// Unit tests for the shared provider error-body normalizer.
//
// See issues/provider-error-body-passthrough. These cover one synthesized error
// object per SDK shape (Mistral, openai APIError, @google/genai ApiError, AWS
// Bedrock ServiceException), plus the non-Error fallback, truncation, the empty
// parsed-body edge case, and the formatProviderError compose helper.

import { APIError } from "openai";
import { describe, expect, it } from "vitest";
import { fauxAssistantMessage } from "../src/providers/faux.ts";
import { formatProviderError, MAX_PROVIDER_ERROR_BODY_CHARS, normalizeProviderError } from "../src/utils/error-body.ts";
import { isRetryableAssistantError } from "../src/utils/retry.ts";

describe("normalizeProviderError", () => {
	it("extracts status and body from a Mistral-shaped error", () => {
		const error = Object.assign(new Error("Mistral request failed"), {
			statusCode: 403,
			body: '{"error":"blocked by gateway WAF"}',
		});

		const norm = normalizeProviderError(error);

		expect(norm.status).toBe(403);
		expect(norm.body).toBe('{"error":"blocked by gateway WAF"}');
		expect(norm.messageCarriesBody).toBe(false);
	});

	it("reads the parsed body off an openai APIError when the message is opaque", () => {
		// makeMessage(status, error, message) yields "<status> status code (no body)"
		// when the parsed body is unparsed, while the body stays on error.error.
		const error = Object.assign(new Error("403 status code (no body)"), {
			status: 403,
			error: { error: "blocked by gateway WAF" },
		});

		const norm = normalizeProviderError(error);

		expect(norm.status).toBe(403);
		expect(norm.body).toBe('{"error":"blocked by gateway WAF"}');
		expect(norm.messageCarriesBody).toBe(false);
	});

	it.each([400, "401", 403, "429", 500, "502", 503, "504", 599])(
		"extracts an HTTP error code from an SDK stream error: %s",
		(code) => {
			const body = { message: "Something went wrong", code };
			const error = new APIError(undefined, body, body.message, undefined);
			const norm = normalizeProviderError(error);

			expect(error.status).toBeUndefined();
			expect(norm.status).toBe(Number(code));
			expect(formatProviderError(norm)).toBe(`${code}: ${JSON.stringify(body)}`);
		},
	);

	it.each([undefined, null, "server_error", "500x", " 500", 200, 600, 500.1, {}, [500]])(
		"does not treat an arbitrary stream error code as an HTTP status: %s",
		(code) => {
			const error = new APIError(undefined, { message: "Something went wrong", code }, undefined, undefined);

			expect(normalizeProviderError(error).status).toBeUndefined();
		},
	);

	it("prefers the HTTP status over a parsed error code", () => {
		const error = new APIError(403, { message: "Permission denied", code: "500" }, undefined, undefined);

		expect(normalizeProviderError(error).status).toBe(403);
	});

	it.each([
		{ code: "429", type: "rate_limit_error", retryable: true },
		{ code: "500", type: "upstream_error", retryable: true },
		{ code: "503", type: "upstream_error", retryable: true },
		{ code: "400", type: "invalid_request_error", retryable: false },
		{ code: "401", type: "authentication_error", retryable: false },
		{ code: "403", type: "permission_error", retryable: false },
		{ code: "429", type: "insufficient_quota", retryable: false },
		{ code: "unknown", type: "unknown", retryable: false },
	])("retains the retry classification for $code/$type", ({ code, type, retryable }) => {
		const error = new APIError(undefined, { message: "Something went wrong", code, type }, undefined, undefined);
		const errorMessage = formatProviderError(normalizeProviderError(error));

		expect(isRetryableAssistantError(fauxAssistantMessage("", { stopReason: "error", errorMessage }))).toBe(
			retryable,
		);
	});

	it("preserves the message when @google/genai already folds the body into it", () => {
		const body = { error: { code: 403, message: "Permission denied" } };
		const error = Object.assign(new Error(JSON.stringify(body)), {
			status: 403,
		});

		const norm = normalizeProviderError(error);

		expect(norm.status).toBe(403);
		expect(norm.messageCarriesBody).toBe(true);
		expect(norm.message).toBe(JSON.stringify(body));
	});

	it("extracts status and body from a Bedrock-shaped ServiceException", () => {
		const error = Object.assign(new Error("UnknownError"), {
			name: "UnknownError",
			$metadata: { httpStatusCode: 403 },
			$response: { statusCode: 403, body: '{"message":"blocked by gateway WAF"}' },
		});

		const norm = normalizeProviderError(error);

		expect(norm.status).toBe(403);
		expect(norm.body).toBe('{"message":"blocked by gateway WAF"}');
		expect(norm.messageCarriesBody).toBe(false);
	});

	it("ignores a Bedrock response stream instead of serializing its internals", () => {
		const error = Object.assign(
			new Error("Invocation of model ID anthropic.claude-opus-5 with on-demand throughput isn't supported."),
			{
				name: "ValidationException",
				$metadata: { httpStatusCode: 400 },
				$response: {
					statusCode: 400,
					body: { pipe: () => undefined, _events: { close: [null, null] } },
				},
			},
		);

		const norm = normalizeProviderError(error);

		expect(norm.status).toBe(400);
		expect(norm.body).toBeUndefined();
		expect(norm.message).toContain("on-demand throughput isn't supported");
		expect(norm.messageCarriesBody).toBe(true);
	});

	it("JSON-stringifies a non-Error thrown value", () => {
		const norm = normalizeProviderError({ reason: "boom" });

		expect(norm.status).toBeUndefined();
		expect(norm.body).toBeUndefined();
		expect(norm.message).toBe('{"reason":"boom"}');
		expect(norm.messageCarriesBody).toBe(false);
	});

	it("treats an empty parsed body object as no body", () => {
		const error = Object.assign(new Error("403 status code (no body)"), {
			status: 403,
			error: {},
		});

		const norm = normalizeProviderError(error);

		expect(norm.body).toBeUndefined();
		expect(norm.messageCarriesBody).toBe(true);
	});

	it("truncates the body at the cap", () => {
		const longBody = "x".repeat(MAX_PROVIDER_ERROR_BODY_CHARS + 50);
		const error = Object.assign(new Error("failed"), {
			statusCode: 500,
			body: longBody,
		});

		const norm = normalizeProviderError(error);

		expect(norm.body).toContain("... [truncated 50 chars]");
		expect(norm.body?.length).toBeLessThan(longBody.length);
	});

	it("sets messageCarriesBody when the message already contains the extracted body", () => {
		const error = Object.assign(new Error("500: upstream exploded"), {
			statusCode: 500,
			body: "upstream exploded",
		});

		const norm = normalizeProviderError(error);

		expect(norm.messageCarriesBody).toBe(true);
	});
});

describe("formatProviderError", () => {
	it("surfaces status and body without a prefix", () => {
		const norm = normalizeProviderError(
			Object.assign(new Error("403 status code (no body)"), {
				status: 403,
				error: { error: "blocked by gateway WAF" },
			}),
		);

		const formatted = formatProviderError(norm);

		expect(formatted).toContain("403");
		expect(formatted).toContain("blocked by gateway WAF");
		expect(formatted).not.toBe("403 status code (no body)");
	});

	it("applies a provider prefix with status and body", () => {
		const norm = normalizeProviderError(
			Object.assign(new Error("403 status code (no body)"), {
				status: 403,
				error: { error: "blocked by gateway WAF" },
			}),
		);

		expect(formatProviderError(norm, "OpenAI API error")).toBe(
			'OpenAI API error (403): {"error":"blocked by gateway WAF"}',
		);
	});

	it("preserves the message (with prefix + status) when it already carries the body", () => {
		const body = JSON.stringify({ error: { message: "Permission denied" } });
		const norm = normalizeProviderError(Object.assign(new Error(body), { status: 403 }));

		expect(formatProviderError(norm, "OpenAI API error")).toBe(`OpenAI API error (403): ${body}`);
	});

	it("returns the bare message for a non-Error value", () => {
		const norm = normalizeProviderError({ reason: "boom" });

		expect(formatProviderError(norm)).toBe('{"reason":"boom"}');
	});
});
