from __future__ import annotations

import fnmatch
import os
from urllib.parse import urlsplit


# DARKBLOOM PATCH: Google APIs need real OAuth Bearer tokens (iron-proxy can't
# mint them from a raw SA JSON), so on our deployment the sandbox entrypoint
# materializes the real GOOGLE_SA_KEY at $GOOGLE_APPLICATION_CREDENTIALS and
# adds *.googleapis.com to NO_PROXY. This module now (a) returns SA credentials
# alongside the http transport so callers can pass them to discovery.build(),
# and (b) honors NO_PROXY so direct httplib2 requests skip iron-proxy.

_GOOGLE_SCOPES_DEFAULT = (
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/documents.readonly",
)


def _proxy_bypass_hosts() -> tuple[str, ...]:
    """Return host patterns (glob-style) that should NOT be routed via HTTPS_PROXY.

    Reads NO_PROXY/no_proxy and accepts both bare hostnames and ``*.example.com``
    style globs. Leading dots (``.example.com``) are normalized to ``*.example.com``.
    """
    raw = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    out: list[str] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("."):
            part = "*" + part
        out.append(part.lower())
    return tuple(out)


def _proxy_info_for_host(host: str | None):
    """Return an httplib2.ProxyInfo for `host`, or None if the host is in NO_PROXY
    or HTTPS_PROXY isn't set.

    httplib2 doesn't natively honor NO_PROXY, so this checks the deny-list
    ourselves before wiring proxy_info. Hostname matching is glob-style
    (case-insensitive). The lazy imports keep workflow modules importable in
    test environments that don't have httplib2 installed.
    """
    import httplib2
    import socks

    proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if not proxy_url:
        return None
    if host:
        host_l = host.lower()
        for pat in _proxy_bypass_hosts():
            if fnmatch.fnmatchcase(host_l, pat):
                return None
    parts = urlsplit(proxy_url)
    return httplib2.ProxyInfo(
        proxy_type=socks.PROXY_TYPE_HTTP,
        proxy_host=parts.hostname,
        proxy_port=parts.port or 8080,
    )


def build_http(host: str | None = None):
    """Build a google-api-python-client HTTP transport.

    By default routes through iron-proxy (HTTPS_PROXY); if `host` matches a
    pattern in NO_PROXY (e.g. ``*.googleapis.com``), goes direct so the
    google-api-python-client's own OAuth flow can authenticate via the SA
    credentials returned by build_credentials().

    Callers that don't pass `host` get proxy routing unconditionally (matches
    upstream's old behavior). Drive/Calendar/Docs services SHOULD pass
    ``host="www.googleapis.com"`` so they bypass iron-proxy.
    """
    import httplib2

    proxy_info = _proxy_info_for_host(host)
    ca_certs = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    return httplib2.Http(proxy_info=proxy_info, ca_certs=ca_certs)


def build_credentials(scopes: tuple[str, ...] = _GOOGLE_SCOPES_DEFAULT):
    """Return google-auth SA credentials loaded from GOOGLE_APPLICATION_CREDENTIALS.

    Returns None when:
      * GOOGLE_APPLICATION_CREDENTIALS isn't set / file missing
      * the file is the upstream mock (client_email==mock@creds.com) — in that
        case there's nothing useful to authenticate as, so fall back to the
        proxy-injection model.
    """
    import json

    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("client_email") == "mock@creds.com":
        return None
    from google.oauth2 import service_account

    return service_account.Credentials.from_service_account_file(path, scopes=list(scopes))
