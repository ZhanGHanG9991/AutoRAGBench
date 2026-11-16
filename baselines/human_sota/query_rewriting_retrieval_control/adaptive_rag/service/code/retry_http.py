"""POST retry for the metering gateway.

On HTTP 5xx, a connection error, or a timeout, retry at most twice
(three attempts). Wait 0.5s, then 1s. Never retry any 4xx. Each attempt
is a real POST, so the gateway counts it toward the 4-call / 32k-token
budget. Do not start a retry when the question clock plus the backoff
would pass the 60s budget minus a 5s margin.

send() must perform one POST and return an httpx.Response whose body is
already read (or still open). This module reads the body before deciding.
"""
import time

import httpx

ATTEMPTS = 3
BACKOFF_S = (0.5, 1.0)
QUESTION_BUDGET_S = 60.0
MARGIN_S = 5.0


def retry_allowed(question_t0, backoff):
    if question_t0 is None:
        return True
    elapsed = time.monotonic() - float(question_t0)
    return elapsed + float(backoff) <= QUESTION_BUDGET_S - MARGIN_S


def post_with_retry(send, question_t0=None):
    """Call send() up to three times. Return the successful response."""
    for attempt in range(ATTEMPTS):
        try:
            response = send()
        except (httpx.TimeoutException, httpx.TransportError):
            if attempt + 1 >= ATTEMPTS or not retry_allowed(question_t0, BACKOFF_S[attempt]):
                raise
            time.sleep(BACKOFF_S[attempt])
            continue
        response.read()
        if 500 <= response.status_code <= 599:
            if attempt + 1 >= ATTEMPTS or not retry_allowed(question_t0, BACKOFF_S[attempt]):
                response.raise_for_status()
            time.sleep(BACKOFF_S[attempt])
            continue
        response.raise_for_status()
        return response
    raise RuntimeError("retry loop ended without a response")
