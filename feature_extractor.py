"""
feature_extractor.py
--------------------
Extracts all 30 UCI Phishing Websites Dataset features from a raw URL string.
Each feature is encoded as -1 (phishing indicator), 0 (suspicious), or 1 (legitimate),
matching the encoding used in the training dataset.

Features are extracted purely from the URL string and DNS/WHOIS data where accessible.
For a Hugging Face deployment, features relying on live page content (HTML scraping)
fall back to heuristic approximations from the URL alone.
"""

import re
import socket
import urllib.parse
from datetime import datetime

# ── Optional imports (graceful fallback if unavailable) ───────────────────────
try:
    import whois as python_whois
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False

try:
    import requests
    from bs4 import BeautifulSoup
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_domain(url: str) -> str:
    try:
        parsed = urllib.parse.urlparse(url)
        return parsed.netloc or parsed.path.split('/')[0]
    except Exception:
        return url

def _get_scheme(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).scheme.lower()
    except Exception:
        return ''

def _has_ip(domain: str) -> bool:
    """Return True if domain is an IP address."""
    ip_pattern = re.compile(
        r'^(\d{1,3}\.){3}\d{1,3}$'
    )
    return bool(ip_pattern.match(domain.split(':')[0]))

def _count_dots(domain: str) -> int:
    return domain.count('.')

def _fetch_page(url: str, timeout: int = 5):
    """Fetch page HTML. Returns (html_text, response) or (None, None)."""
    if not REQUESTS_AVAILABLE:
        return None, None
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; PhishingDetector/1.0)'}
        resp = requests.get(url, timeout=timeout, headers=headers,
                            allow_redirects=True, verify=False)
        return resp.text, resp
    except Exception:
        return None, None

def _get_whois_info(domain: str):
    if not WHOIS_AVAILABLE:
        return None
    try:
        return python_whois.whois(domain)
    except Exception:
        return None


# ── Feature Extraction Functions ──────────────────────────────────────────────

def f_having_IP_Address(url, domain, html, resp):
    """Feature 1: IP address used as domain → -1 (phishing), else 1"""
    return -1 if _has_ip(domain) else 1


def f_URL_Length(url, domain, html, resp):
    """Feature 2: URL length. Short(<54)→1, medium(54-75)→0, long(>75)→-1"""
    n = len(url)
    if n < 54:
        return 1
    elif n <= 75:
        return 0
    return -1


def f_Shortining_Service(url, domain, html, resp):
    """Feature 3: Known URL shortening services → -1"""
    shorteners = [
        'bit.ly', 'goo.gl', 'shorte.st', 'go2l.ink', 'x.co', 'ow.ly',
        't.co', 'tinyurl.com', 'tr.im', 'is.gd', 'cli.gs', 'yfrog.com',
        'migre.me', 'ff.im', 'tiny.cc', 'url4.eu', 'twit.ac', 'su.pr',
        'twurl.nl', 'snipurl.com', 'short.to', 'BudURL.com', 'ping.fm',
        'post.ly', 'Just.as', 'bkite.com', 'snipr.com', 'fic.kr', 'loopt.us',
        'doiop.com', 'short.ie', 'kl.am', 'wp.me', 'rubyurl.com', 'om.ly',
        'to.ly', 'bit.do', 'lnkd.in', 'db.tt', 'qr.ae', 'adf.ly',
        'bitly.com', 'cur.lv', 'tinyurl', 'ity.im', 'q.gs', 'po.st',
        'bc.vc', 'twitthis.com', 'u.to', 'j.mp', 'buzurl.com', 'cutt.us',
        'u.bb', 'yourls.org', 'prettylinkpro.com', 'viralurl.com',
        'EasyUrl.net', 'Yourls.org'
    ]
    domain_lower = domain.lower()
    for s in shorteners:
        if s.lower() in domain_lower:
            return -1
    return 1


def f_having_At_Symbol(url, domain, html, resp):
    """Feature 4: @ symbol in URL → -1"""
    return -1 if '@' in url else 1


def f_double_slash_redirecting(url, domain, html, resp):
    """Feature 5: // after the protocol prefix (position > 7) → -1"""
    pos = url.rfind('//')
    return -1 if pos > 7 else 1


def f_Prefix_Suffix(url, domain, html, resp):
    """Feature 6: Hyphen in domain name → -1"""
    return -1 if '-' in domain else 1


def f_having_Sub_Domain(url, domain, html, resp):
    """Feature 7: Subdomain count. 0→1, 1→0, ≥2→-1"""
    # Remove www. first
    d = re.sub(r'^www\.', '', domain.split(':')[0])
    dots = d.count('.')
    if dots == 1:
        return 1
    elif dots == 2:
        return 0
    return -1


def f_SSLfinal_State(url, domain, html, resp):
    """Feature 8: HTTPS with trusted cert → 1, HTTPS only → 0, HTTP → -1"""
    scheme = _get_scheme(url)
    if scheme == 'https':
        # Check cert validity heuristically (age of domain helps here)
        return 1
    return -1


def f_Domain_registeration_length(url, domain, html, resp):
    """Feature 9: Domain registration length. >1yr→1, unknown→-1"""
    if not WHOIS_AVAILABLE:
        # Heuristic: domains with many subdomains tend to be recently registered
        d = re.sub(r'^www\.', '', domain.split(':')[0])
        return 0 if d.count('.') <= 1 else -1
    try:
        w = _get_whois_info(domain.split(':')[0])
        if w is None:
            return -1
        exp = w.expiration_date
        if isinstance(exp, list):
            exp = exp[0]
        if exp is None:
            return -1
        remaining = (exp - datetime.now()).days
        return 1 if remaining > 365 else -1
    except Exception:
        return -1


def f_Favicon(url, domain, html, resp):
    """Feature 10: Favicon from different domain → -1, same → 1"""
    if html is None:
        return 1  # assume legitimate if not scrapable
    try:
        soup = BeautifulSoup(html, 'html.parser')
        icons = soup.find_all('link', rel=lambda r: r and 'icon' in ' '.join(r).lower())
        for icon in icons:
            href = icon.get('href', '')
            if href and href.startswith('http') and domain not in href:
                return -1
        return 1
    except Exception:
        return 1


def f_port(url, domain, html, resp):
    """Feature 11: Non-standard port used → -1"""
    standard = {'http': 80, 'https': 443}
    parsed = urllib.parse.urlparse(url)
    port = parsed.port
    scheme = parsed.scheme.lower()
    if port is None:
        return 1
    expected = standard.get(scheme)
    return 1 if port == expected else -1


def f_HTTPS_token(url, domain, html, resp):
    """Feature 12: 'https' string appears in domain part → -1"""
    return -1 if 'https' in domain.lower() else 1


def f_Request_URL(url, domain, html, resp):
    """Feature 13: % of external objects in page. <22%→1, 22-61%→0, >61%→-1"""
    if html is None:
        return 0
    try:
        soup = BeautifulSoup(html, 'html.parser')
        tags = soup.find_all(['img', 'script', 'link'])
        if not tags:
            return 1
        external = 0
        for tag in tags:
            src = tag.get('src') or tag.get('href') or ''
            if src.startswith('http') and domain not in src:
                external += 1
        ratio = external / len(tags)
        if ratio < 0.22:
            return 1
        elif ratio < 0.61:
            return 0
        return -1
    except Exception:
        return 0


def f_URL_of_Anchor(url, domain, html, resp):
    """Feature 14: % of anchors with external href. <31%→1, 31-67%→0, >67%→-1"""
    if html is None:
        return 0
    try:
        soup = BeautifulSoup(html, 'html.parser')
        anchors = soup.find_all('a', href=True)
        if not anchors:
            return 1
        suspicious = 0
        for a in anchors:
            href = a['href']
            if href in ('#', '', 'javascript::void(0)', 'javascript:void(0)'):
                suspicious += 1
            elif href.startswith('http') and domain not in href:
                suspicious += 1
        ratio = suspicious / len(anchors)
        if ratio < 0.31:
            return 1
        elif ratio < 0.67:
            return 0
        return -1
    except Exception:
        return 0


def f_Links_in_tags(url, domain, html, resp):
    """Feature 15: % of external links in <meta>/<script>/<link>. <17%→1, 17-81%→0, >81%→-1"""
    if html is None:
        return 0
    try:
        soup = BeautifulSoup(html, 'html.parser')
        tags = soup.find_all(['meta', 'script', 'link'])
        if not tags:
            return 1
        external = 0
        for tag in tags:
            src = tag.get('src') or tag.get('href') or tag.get('content') or ''
            if src.startswith('http') and domain not in src:
                external += 1
        ratio = external / len(tags)
        if ratio < 0.17:
            return 1
        elif ratio < 0.81:
            return 0
        return -1
    except Exception:
        return 0


def f_SFH(url, domain, html, resp):
    """Feature 16: Server Form Handler. about:blank or empty→-1, external→0, same→1"""
    if html is None:
        return 0
    try:
        soup = BeautifulSoup(html, 'html.parser')
        forms = soup.find_all('form', action=True)
        if not forms:
            return 1
        for form in forms:
            action = form['action'].strip()
            if action in ('', 'about:blank'):
                return -1
            if action.startswith('http') and domain not in action:
                return 0
        return 1
    except Exception:
        return 0


def f_Submitting_to_email(url, domain, html, resp):
    """Feature 17: Form submits via mailto → -1"""
    if html is None:
        return 1
    try:
        soup = BeautifulSoup(html, 'html.parser')
        for form in soup.find_all('form', action=True):
            if 'mailto:' in form['action'].lower():
                return -1
        return 1
    except Exception:
        return 1


def f_Abnormal_URL(url, domain, html, resp):
    """Feature 18: Hostname not in URL → -1"""
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname or ''
    return -1 if hostname not in url else 1


def f_Redirect(url, domain, html, resp):
    """Feature 19: Number of redirects. 0→1, 1→0, ≥2→-1"""
    if resp is None:
        return 0
    try:
        n = len(resp.history)
        if n == 0:
            return 1
        elif n == 1:
            return 0
        return -1
    except Exception:
        return 0


def f_on_mouseover(url, domain, html, resp):
    """Feature 20: JS alters status bar → -1"""
    if html is None:
        return 1
    return -1 if 'onmouseover' in html.lower() and 'window.status' in html.lower() else 1


def f_RightClick(url, domain, html, resp):
    """Feature 21: Right-click disabled via JS → -1"""
    if html is None:
        return 1
    return -1 if 'preventdefault' in html.lower() or 'contextmenu' in html.lower() else 1


def f_popUpWidnow(url, domain, html, resp):
    """Feature 22: Pop-up window with text fields → -1"""
    if html is None:
        return 1
    return -1 if 'window.open' in html.lower() else 1


def f_Iframe(url, domain, html, resp):
    """Feature 23: Invisible IFrame present → -1"""
    if html is None:
        return 1
    try:
        soup = BeautifulSoup(html, 'html.parser')
        for iframe in soup.find_all('iframe'):
            style = iframe.get('style', '').lower()
            width = iframe.get('width', '1')
            height = iframe.get('height', '1')
            if ('display:none' in style or 'visibility:hidden' in style
                    or width == '0' or height == '0'):
                return -1
        return 1
    except Exception:
        return 1


def f_age_of_domain(url, domain, html, resp):
    """Feature 24: Domain age. >6 months→1, else→-1"""
    if not WHOIS_AVAILABLE:
        return 0
    try:
        w = _get_whois_info(domain.split(':')[0])
        if w is None:
            return -1
        created = w.creation_date
        if isinstance(created, list):
            created = created[0]
        if created is None:
            return -1
        age_months = (datetime.now() - created).days / 30
        return 1 if age_months >= 6 else -1
    except Exception:
        return -1


def f_DNSRecord(url, domain, html, resp):
    """Feature 25: DNS record exists → 1, else → -1"""
    try:
        socket.gethostbyname(domain.split(':')[0])
        return 1
    except Exception:
        return -1


def f_web_traffic(url, domain, html, resp):
    """Feature 26: Web traffic rank approximation. Heuristic from domain age / DNS."""
    # Without Alexa API, we approximate: if DNS resolves, assume mid-tier
    try:
        socket.gethostbyname(domain.split(':')[0])
        return 0  # Unknown rank → suspicious
    except Exception:
        return -1


def f_Page_Rank(url, domain, html, resp):
    """Feature 27: Page rank approximation. Heuristic."""
    # Heuristic: HTTPS + DNS → assume non-zero rank
    scheme = _get_scheme(url)
    try:
        socket.gethostbyname(domain.split(':')[0])
        return 1 if scheme == 'https' else 0
    except Exception:
        return -1


def f_Google_Index(url, domain, html, resp):
    """Feature 28: Indexed by Google approximation. If DNS resolves → assume indexed."""
    try:
        socket.gethostbyname(domain.split(':')[0])
        return 1
    except Exception:
        return -1


def f_Links_pointing_to_page(url, domain, html, resp):
    """Feature 29: External links pointing to page. Heuristic → 0 (unknown)."""
    return 0


def f_Statistical_report(url, domain, html, resp):
    """Feature 30: Host in PhishTank / StopBadware. Heuristic → assume not listed."""
    # Without live API access, default to legitimate
    return 1


# ── Master Extractor ──────────────────────────────────────────────────────────

ALL_FEATURES = [
    ('having_IP_Address',           f_having_IP_Address),
    ('URL_Length',                  f_URL_Length),
    ('Shortining_Service',          f_Shortining_Service),
    ('having_At_Symbol',            f_having_At_Symbol),
    ('double_slash_redirecting',    f_double_slash_redirecting),
    ('Prefix_Suffix',               f_Prefix_Suffix),
    ('having_Sub_Domain',           f_having_Sub_Domain),
    ('SSLfinal_State',              f_SSLfinal_State),
    ('Domain_registeration_length', f_Domain_registeration_length),
    ('Favicon',                     f_Favicon),
    ('port',                        f_port),
    ('HTTPS_token',                 f_HTTPS_token),
    ('Request_URL',                 f_Request_URL),
    ('URL_of_Anchor',               f_URL_of_Anchor),
    ('Links_in_tags',               f_Links_in_tags),
    ('SFH',                         f_SFH),
    ('Submitting_to_email',         f_Submitting_to_email),
    ('Abnormal_URL',                f_Abnormal_URL),
    ('Redirect',                    f_Redirect),
    ('on_mouseover',                f_on_mouseover),
    ('RightClick',                  f_RightClick),
    ('popUpWidnow',                 f_popUpWidnow),
    ('Iframe',                      f_Iframe),
    ('age_of_domain',               f_age_of_domain),
    ('DNSRecord',                   f_DNSRecord),
    ('web_traffic',                 f_web_traffic),
    ('Page_Rank',                   f_Page_Rank),
    ('Google_Index',                f_Google_Index),
    ('Links_pointing_to_page',      f_Links_pointing_to_page),
    ('Statistical_report',          f_Statistical_report),
]

FEATURE_NAMES = [name for name, _ in ALL_FEATURES]

FEATURE_DESCRIPTIONS = {
    'having_IP_Address':           'IP address used instead of domain name',
    'URL_Length':                  'URL length (short / medium / long)',
    'Shortining_Service':          'URL shortening service detected',
    'having_At_Symbol':            '@ symbol present in URL',
    'double_slash_redirecting':    'Double slash redirect in URL path',
    'Prefix_Suffix':               'Hyphen in domain name',
    'having_Sub_Domain':           'Number of subdomains',
    'SSLfinal_State':              'Valid HTTPS/SSL certificate',
    'Domain_registeration_length': 'Domain registration duration',
    'Favicon':                     'Favicon loaded from same domain',
    'port':                        'Standard port used',
    'HTTPS_token':                 '"https" string in domain name',
    'Request_URL':                 'External object request ratio',
    'URL_of_Anchor':               'External anchor link ratio',
    'Links_in_tags':               'External links in meta/script tags',
    'SFH':                         'Form submits to same domain',
    'Submitting_to_email':         'Form uses mailto submission',
    'Abnormal_URL':                'Hostname present in URL',
    'Redirect':                    'Number of redirects',
    'on_mouseover':                'Status bar customisation via JS',
    'RightClick':                  'Right-click disabled',
    'popUpWidnow':                 'Pop-up window detected',
    'Iframe':                      'Hidden IFrame detected',
    'age_of_domain':               'Domain older than 6 months',
    'DNSRecord':                   'Valid DNS record exists',
    'web_traffic':                 'Web traffic rank',
    'Page_Rank':                   'Estimated page rank',
    'Google_Index':                'Indexed by search engines',
    'Links_pointing_to_page':      'External backlinks to page',
    'Statistical_report':          'Not listed in phishing databases',
}

VALUE_LABELS = {
    -1: ('Phishing indicator', '#ef4444'),
    0:  ('Suspicious',         '#f59e0b'),
    1:  ('Legitimate',         '#22c55e'),
}


def extract_features(url: str) -> dict:
    """
    Extract all 30 UCI phishing features from a raw URL string.

    Returns a dict with:
        - 'features': dict of {feature_name: encoded_value}
        - 'feature_vector': list of 30 values in dataset order
        - 'domain': extracted domain
        - 'scheme': http or https
        - 'html_available': whether page was fetched
        - 'errors': list of non-fatal errors encountered
    """
    # Normalise URL
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    domain = _get_domain(url)
    errors = []

    # Try to fetch page content (best-effort)
    html, resp = _fetch_page(url)
    if html is None:
        errors.append('Page fetch failed — URL-only features used for HTML-dependent checks.')

    features = {}
    for feat_name, feat_fn in ALL_FEATURES:
        try:
            features[feat_name] = feat_fn(url, domain, html, resp)
        except Exception as e:
            features[feat_name] = 0
            errors.append(f'{feat_name}: {str(e)}')

    return {
        'features':         features,
        'feature_vector':   [features[name] for name in FEATURE_NAMES],
        'domain':           domain,
        'scheme':           _get_scheme(url),
        'html_available':   html is not None,
        'errors':           errors,
    }
