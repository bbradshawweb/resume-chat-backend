import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError

import main as backend


class ChatErrorTests(unittest.TestCase):
    def setUp(self):
        backend.app.config["TESTING"] = True
        backend.rate_limit_state.clear()
        self.client = backend.app.test_client()
        self.create = Mock()
        self.upstream = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))

    def post_chat(self, **payload):
        backend.rate_limit_state.clear()
        with patch.object(backend, "get_openai_client", return_value=self.upstream):
            return self.client.post(
                "/chat", json={"message": "What is your current role?", **payload},
                headers={"Origin": "https://bradenbradshaw.com"},
            )

    def provider_error(self, status, code=None, error_type="invalid_request_error", param=None):
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(status, request=request, headers={"x-request-id": "req-unit-test"})
        return APIStatusError(
            "Provider detail with sk-test-secret-that-must-not-leak",
            response=response,
            body={"error": {"code": code, "type": error_type, "param": param}},
        )

    def test_provider_errors_keep_diagnostics_out_of_public_response(self):
        cases = [
            (401, "invalid_api_key", 503, "upstream_access_denied"),
            (403, "permission_denied", 503, "upstream_access_denied"),
            (404, "model_not_found", 503, "upstream_model_unavailable"),
            (400, "unsupported_parameter", 502, "upstream_request_rejected"),
            (422, None, 502, "upstream_request_rejected"),
            (500, "internal_error", 502, "upstream_service_error"),
            (503, "server_is_overloaded", 502, "upstream_service_error"),
        ]
        for upstream_status, provider_code, expected_status, expected_code in cases:
            with self.subTest(upstream_status=upstream_status):
                self.create.side_effect = self.provider_error(upstream_status, provider_code, param="max_tokens")
                with self.assertLogs(backend.logger, level="INFO") as logs:
                    response = self.post_chat()
                self.assertEqual(response.status_code, expected_status)
                self.assertEqual(response.json["code"], expected_code)
                self.assertTrue(response.json["message"])
                self.assertEqual(response.headers["Access-Control-Allow-Origin"], "https://bradenbradshaw.com")
                self.assertIn("req-unit-test", "\n".join(logs.output))
                self.assertIn('"parameter": "max_tokens"', "\n".join(logs.output))
                self.assertNotIn("sk-test-secret", response.get_data(as_text=True))
                self.assertNotIn("sk-test-secret", "\n".join(logs.output))
                self.assertNotIn("Retry-After", response.headers)

    def test_quota_failures_do_not_tell_visitors_to_retry_as_a_rate_limit(self):
        for code in (
            "insufficient_quota", "billing_hard_limit_reached", "billing_not_active",
            "organization_spend_limit_exceeded", "project_spend_limit_exceeded",
            "organization_usage_limit_exceeded",
        ):
            with self.subTest(code=code):
                self.create.side_effect = self.provider_error(429, code)
                response = self.post_chat()
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json["code"], "upstream_quota_exhausted")
                self.assertNotIn("retry_after", response.json)
                self.assertNotIn("Retry-After", response.headers)

    def test_quota_type_is_recognized_without_code(self):
        self.create.side_effect = self.provider_error(429, error_type="insufficient_quota")
        self.assertEqual(self.post_chat().json["code"], "upstream_quota_exhausted")

    def test_temporary_rate_limit_retains_retry_after(self):
        self.create.side_effect = self.provider_error(429, "rate_limit_exceeded", "rate_limit_error")
        response = self.post_chat()
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json["code"], "upstream_rate_limited")
        self.assertEqual(response.headers["Retry-After"], "20")

    def test_connection_and_timeout_errors(self):
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        for error in (APIConnectionError(request=request), APITimeoutError(request=request)):
            with self.subTest(error=type(error).__name__):
                self.create.side_effect = error
                response = self.post_chat()
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json["code"], "upstream_unavailable")

    def test_client_initialization_error_is_caught(self):
        with patch.object(backend, "get_openai_client", side_effect=RuntimeError("Initialization failed")):
            response = self.client.post("/chat", json={"message": "Hello"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json["code"], "server_error")
        self.assertNotIn("Initialization failed", response.get_data(as_text=True))

    def test_missing_key_and_health_check(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "  "}):
            self.assertFalse(self.client.get("/health").json["openai_configured"])
            response = self.client.post("/chat", json={"message": "Hello"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json["code"], "missing_api_key")
        self.create.assert_not_called()

    def test_empty_completions_and_empty_text_are_errors(self):
        for choices in ([], [SimpleNamespace(message=SimpleNamespace(content=" "))]):
            with self.subTest(choices=choices):
                self.create.return_value = SimpleNamespace(choices=choices)
                response = self.post_chat()
                self.assertEqual(response.status_code, 502)
                self.assertEqual(response.json["code"], "upstream_empty_response")

    def test_success_preserves_conversation_context_and_project_prompts(self):
        answer = "I lead operations and marketing data science at AlterMe."
        self.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))])
        response = self.post_chat(
            history=[{"role": "user", "content": "Tell me about Salesforce"},
                     {"role": "assistant", "content": "I built a workcenter."},
                     {"role": "user", "content": "What is your current role?"}],
            page_context={"selected_project": "Salesforce Workcenter"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["response"], answer)
        self.assertEqual(response.json["history_used"], 2)
        self.assertIsInstance(response.json["suggested_followups"], list)
        messages = self.create.call_args.kwargs["messages"]
        self.assertIn("Salesforce Workcenter", messages[0]["content"])
        self.assertEqual(sum(m["content"] == "What is your current role?" for m in messages), 1)

    def test_contact_handoff_remains_available(self):
        answer = "Would you like me to pass this conversation along to Braden?"
        self.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))])
        with patch.object(backend, "deliver_notification_email") as send_email:
            response = self.post_chat(message="I want to hire you. Can we schedule an interview?")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["contact_handoff"]["enabled"])
        self.assertEqual(response.json["suggested_followups"], [])
        send_email.assert_not_called()

    def test_luna_request_uses_compatible_parameters_without_reasoning_overhead(self):
        answer = "I build revenue operations systems."
        self.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))])
        with patch.object(backend, "MODEL", "gpt-5.6-luna"):
            response = self.post_chat()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["model"], "gpt-5.6-luna")
        parameters = self.create.call_args.kwargs
        self.assertEqual(parameters["max_completion_tokens"], 700)
        self.assertEqual(parameters["reasoning_effort"], "none")
        self.assertNotIn("temperature", parameters)
        self.assertNotIn("max_tokens", parameters)

    def test_non_reasoning_model_override_does_not_receive_reasoning_parameters(self):
        self.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Hello"))])
        with patch.object(backend, "MODEL", "gpt-4.1-mini"):
            response = self.post_chat()
        self.assertEqual(response.status_code, 200)
        parameters = self.create.call_args.kwargs
        self.assertEqual(parameters["temperature"], 0.35)
        self.assertEqual(parameters["max_completion_tokens"], 700)
        self.assertNotIn("reasoning_effort", parameters)
        self.assertNotIn("max_tokens", parameters)

    def test_v2_alias_has_the_same_error_contract(self):
        self.create.side_effect = self.provider_error(401, "invalid_api_key")
        with patch.object(backend, "get_openai_client", return_value=self.upstream):
            response = self.client.post("/chat/v2", json={"message": "Hello"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json["code"], "upstream_access_denied")

    def test_non_object_json_is_rejected_before_calling_provider(self):
        response = self.client.post("/chat", json=["hello"])
        self.assertEqual(response.status_code, 400)
        self.create.assert_not_called()

    def test_contact_rejects_non_object_json(self):
        response = self.client.post("/contact", json=["invalid"])
        self.assertEqual(response.status_code, 400)

    def test_assistant_email_is_not_mistaken_for_visitor_contact(self):
        signal = backend.score_conversation_strength(
            "What tools do you use?", "My email is bradshaw.braden@gmail.com.", []
        )
        self.assertEqual(signal["visitor_emails"], [])
        self.assertFalse(backend.should_offer_contact_handoff(signal))

    def test_call_review_email_automation_and_contract_research_do_not_trigger_contact(self):
        for question in (
            "How do your call reviews work?",
            "What email automation did you build?",
            "Tell me about your university research contract.",
        ):
            with self.subTest(question=question):
                signal = backend.score_conversation_strength(question, "I built custom systems.", [])
                self.assertFalse(backend.should_offer_contact_handoff(signal))

    def test_actual_contact_intent_and_visitor_email_are_preserved(self):
        signal = backend.score_conversation_strength(
            "Can we schedule a call? Email me at visitor@example.com.", "Happy to discuss.", []
        )
        self.assertEqual(signal["visitor_emails"], ["visitor@example.com"])
        self.assertTrue(backend.should_offer_contact_handoff(signal))

    def test_experiment_followups_use_the_lifecycle_project(self):
        prompts = backend.pick_followups("Tell me about your Salesforce A/B testing reports.", "I built a custom MCP.")
        self.assertEqual(prompts, backend.profile_context["follow_up_prompts"]["lifecycle_experimentation"])

    def test_reports_do_not_accidentally_match_the_rep_keyword(self):
        prompts = backend.pick_followups("What reports did you create?", "I use statistical modeling.")
        self.assertEqual(prompts, backend.profile_context["follow_up_prompts"]["data_science"][:3])

    def test_failed_email_delivery_does_not_consume_contact_cooldown(self):
        backend.notification_state.clear()
        with (
            patch.object(backend, "NOTIFICATION_ENABLED", True),
            patch.object(backend, "NOTIFICATION_FROM_EMAIL", "portfolio@example.com"),
            patch.object(backend, "notification_configured", return_value=True),
            patch.object(backend, "deliver_notification_email", return_value=False) as deliver,
        ):
            first = self.client.post("/contact", json={"email": "visitor@example.com", "summary": "Interested in lifecycle experiments."})
            second = self.client.post("/contact", json={"email": "visitor@example.com", "summary": "Interested in lifecycle experiments."})
        self.assertEqual(first.status_code, 502)
        self.assertEqual(second.status_code, 502)
        self.assertEqual(deliver.call_count, 2)
        self.assertEqual(backend.notification_state, {})

    def test_contact_validation_prevents_delivery_with_invalid_email(self):
        with (
            patch.object(backend, "notification_configured", return_value=True),
            patch.object(backend, "deliver_notification_email") as deliver,
        ):
            response = self.client.post("/contact", json={"email": "not-an-email"})
        self.assertEqual(response.status_code, 400)
        deliver.assert_not_called()


if __name__ == "__main__":
    unittest.main()
