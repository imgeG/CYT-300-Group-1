import re, html, unicodedata
import numpy as np
import pandas as pd
from functools import lru_cache
from urllib.parse import urlparse
from scipy.sparse import hstack, csr_matrix

KEY = ['sender', 'receiver', 'subject', 'body']     # columns that define "the same email"

def msg_to_row(msg):
    """One parsed email -> sender / receiver / subject / body (None if unreadable)."""
    try:
        part = msg.get_body(preferencelist=('plain', 'html'))      # prefer plain text, else HTML
        body = part.get_content() if part is not None else ''
        return {'sender': str(msg.get('From', '') or ''), 'receiver': str(msg.get('To', '') or ''),
                'subject': str(msg.get('Subject', '') or ''), 'body': str(body)}
    except Exception:
        return None

# Cleaning

def clean(d, verbose=True):
    """Drop rows with no sender/label, fill missing text, remove contradictory and duplicate emails."""
    before = len(d)
    d = d.dropna(subset=['sender', 'label']).copy()
    for c in KEY:
        d[c] = d[c].fillna('').astype(str)                     # missing text -> empty string
    d['label'] = d['label'].astype(int)                        # 0 = legit, 1 = phishing/spam
    d = d[d.groupby(KEY)['label'].transform('nunique') == 1]   # same email with both labels -> drop
    d = d.drop_duplicates(subset=KEY)                          # keep one copy of repeated emails
    if verbose:
        print(f"clean(): {before} -> {len(d)} rows")
    return d


# Address and text helpers

ADDR_RE = re.compile(r'[\w.+-]+@[\w-]+(?:\.[\w-]+)+')     # an email address
FREE_MAIL = {'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com',
             'msn.com', 'live.com', 'mail.com', 'yandex.ru', 'gmx.com'}

def first_address(s):
    """First email address in a field, lowercased ('' if none)."""
    m = ADDR_RE.search(s)
    return m.group(0).lower() if m else ''

def display_name(s):
    """Name before the address: 'PayPal Support <x@y.com>' -> 'PayPal Support'."""
    return s.split('<')[0].strip().strip('"').strip() if '<' in s else ''

URL_RE   = re.compile(r'https?://\S+|www\.\S+')
TAG_RE   = re.compile(r'<[^>]+>')
BLOCK_RE = re.compile(r'(?is)<(script|style).*?</\1>')

def clean_text(s, max_len=5000):
    """Prepare text for TF-IDF: strip HTML, decode entities, replace links/addresses with tokens, lowercase."""
    s = s[:20000]                          # cap huge bodies (speed)
    s = BLOCK_RE.sub(' ', s)               # remove <script>/<style> blocks
    s = TAG_RE.sub(' ', s)                 # remove HTML tags
    s = html.unescape(s)                   # &amp; -> &
    s = URL_RE.sub(' urltoken ', s)        # each link -> one placeholder word
    s = ADDR_RE.sub(' emailtoken ', s)     # each address -> one placeholder word
    return re.sub(r'\s+', ' ', s).strip().lower()[:max_len]

# Unicode tricks: invisible characters and lookalike letters
INVIS = re.compile(r'[\u200b-\u200f\u2060\ufeff\ufe00-\ufe0f\U000e0100-\U000e01ef]')

def norm_text(s):
    """Remove invisible characters and fold lookalike forms (bold/fancy letters -> plain)."""
    return INVIS.sub('', unicodedata.normalize('NFKC', s))

def mixed_script(s):
    """1 if letters from 2+ alphabets are mixed (e.g. Latin text with a Cyrillic 'а')."""
    scripts = {unicodedata.name(c, '').split(' ')[0] for c in s if c.isalpha()}
    return int(len(scripts & {'LATIN', 'CYRILLIC', 'GREEK'}) >= 2)


# Phishing indicators (fixed rules, nothing is learned)
# VirusTotal answers "is this link KNOWN to be bad"; these answer "does this email LOOK deceptive".

BRANDS = {
    'paypal': {'paypal.com'},
    'microsoft': {'microsoft.com', 'live.com', 'outlook.com', 'office.com', 'office365.com', 'microsoftonline.com'},
    'google': {'google.com', 'gmail.com', 'googlemail.com'},
    'amazon': {'amazon.com', 'amazon.ca', 'amazonaws.com'},
    'apple': {'apple.com', 'icloud.com'},
    'netflix': {'netflix.com'},
    'facebook': {'facebook.com', 'fb.com', 'facebookmail.com'},
    'instagram': {'instagram.com'},
    'linkedin': {'linkedin.com'},
    'dropbox': {'dropbox.com'},
    'docusign': {'docusign.com', 'docusign.net'},
    'fedex': {'fedex.com'},
    'chase': {'chase.com'},
    'wellsfargo': {'wellsfargo.com'},
    'bankofamerica': {'bankofamerica.com'},
    'citibank': {'citibank.com', 'citi.com'},
    'americanexpress': {'americanexpress.com', 'aexp.com'},
    'ebay': {'ebay.com'},
}
MIN_FUZZY_LEN = 5      # no fuzzy matching for very short brand names (too many false matches)
SUSP_TLDS  = {'xyz', 'top', 'click', 'buzz', 'tk', 'ml', 'ga', 'cf', 'gq', 'work', 'link', 'icu',
              'cfd', 'sbs', 'rest', 'zip', 'mov', 'ru', 'cn'}
SHORTENERS = {'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly', 'is.gd', 'buff.ly',
              'rebrand.ly', 'cutt.ly', 'tiny.cc'}
TWO_PART_SUFFIX = {'co.uk', 'com.au', 'co.jp', 'com.br', 'co.in', 'org.uk', 'com.cn', 'co.za', 'com.mx'}
LEET = str.maketrans({'0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's', '7': 't', '$': 's', '@': 'a'})

def norm_label(s):
    """Undo lookalike tricks: paypa1 -> paypal, rnicrosoft -> microsoft."""
    return s.translate(LEET).replace('rn', 'm').replace('vv', 'w')

def lev(a, b):
    """Levenshtein distance: single-character edits to turn a into b (paypall vs paypal = 1)."""
    if abs(len(a) - len(b)) > 2:
        return 3
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def reg_domain(host):
    """Registrable domain: mail.evil.co.uk -> evil.co.uk, login.paypal.com -> paypal.com."""
    p = host.lower().strip('.').split('.')
    if len(p) >= 3 and '.'.join(p[-2:]) in TWO_PART_SUFFIX:
        return '.'.join(p[-3:])
    return '.'.join(p[-2:]) if len(p) >= 2 else host

@lru_cache(maxsize=None)
def domain_risk(host):
    """Flags: typo (paypall.com), leet (paypa1.com), brand in domain (paypal-support.com),
    brand in subdomain (paypal.com.secure-login.ru)."""
    host = host.lower().strip('.')
    reg  = reg_domain(host)
    sld  = reg.split('.')[0]
    sub  = host[:-len(reg)].strip('.') if host.endswith(reg) else ''
    tokens = {sld} | set(sld.split('-'))
    typo = leet = in_dom = in_sub = 0
    for brand, official in BRANDS.items():
        if reg in official:
            continue                                       # the real brand domain is not a squat
        for t in tokens:
            if not t:
                continue
            if t.startswith(brand):                        # 'paypal-support' yes; 'purchase', 'pineapple' no
                in_dom = 1
            elif brand in t:                               # brand inside another word ('onmicrosoft', 'purchase'):
                continue                                   # not a lookalike, so skip the typo/leet check
            elif len(brand) >= MIN_FUZZY_LEN:
                n = norm_label(t)
                if n == brand:
                    leet = 1
                elif lev(n, brand) <= (1 if len(brand) <= 7 else 2):
                    typo = 1
        if sub and any(p.startswith(brand) for p in re.split(r'[.-]', sub)):
            in_sub = 1
    return typo, leet, in_dom, in_sub

def sender_risk(domain, name):
    """Sender impersonation flags, including a brand in the display name: 'PayPal <x@random.com>'."""
    typo, leet, in_dom, in_sub = domain_risk(domain)
    reg = reg_domain(domain)
    name_key = re.sub(r'[^a-z0-9]', '', name.lower())
    disp = int(any(b in name_key and reg not in off for b, off in BRANDS.items()))
    return {'s_typosquat': typo, 's_leet': leet, 's_brand_in_domain': in_dom,
            's_brand_in_subdomain': in_sub, 's_display_brand_mismatch': disp,
            's_punycode': int('xn--' in domain),
            's_nonascii': int(any(ord(c) > 127 for c in domain)),
            's_susp_tld': int(domain.split('.')[-1] in SUSP_TLDS)}

RAW_URL_RE = re.compile(r'https?://[^\s<>"\')]+', re.I)
HREF_RE    = re.compile(r'(?is)<a\s[^>]*href\s*=\s*["\']?([^"\'\s>]+)[^>]*>(.*?)</a>')
DOMAINISH  = re.compile(r'\b[a-z0-9-]+(?:\.[a-z0-9-]+)+\b')
IP_RE      = re.compile(r'^\d{1,3}(?:\.\d{1,3}){3}$')

def host_of(u):
    try:
        return (urlparse(u).hostname or '').lower()
    except ValueError:
        return ''

def netloc_of(u):
    try:
        return urlparse(u).netloc
    except ValueError:
        return ''

def url_risk(body):
    """Structural link checks (VirusTotal handles link reputation)."""
    body  = body[:20000]
    urls  = RAW_URL_RE.findall(body)
    hosts = [h for h in (host_of(u) for u in urls) if h]
    out = dict(u_count=len(urls), u_ip=0, u_shortener=0, u_at=0, u_anchor_mismatch=0)
    if not hosts:
        return out
    out['u_ip']        = int(any(IP_RE.match(h) for h in hosts))               # link to a bare IP
    out['u_shortener'] = int(any(reg_domain(h) in SHORTENERS for h in hosts))  # hidden destination
    out['u_at']        = int(any('@' in netloc_of(u) for u in urls))          # http://paypal.com@evil.com
    for href, text in HREF_RE.findall(body):                                    # link text shows one domain,
        shown = DOMAINISH.findall(TAG_RE.sub(' ', text).lower())                # link goes to another
        h = host_of(href)
        if shown and h and reg_domain(shown[0]) != reg_domain(h):
            out['u_anchor_mismatch'] = 1
            break
    return out

URGENCY_RE = r'\b(?:urgent|immediately|within 24 hours|suspended|final notice|act now|expire[sd]?|limited time|verify your|confirm your)\b'
CRED_RE    = r'\b(?:password|log ?in|sign ?in|username|social security|ssn|account number|credit card|card number|pin)\b'


# Feature builder

def build_features(d, verbose=True):
    """Parse addresses and compute address, text and phishing-indicator features."""
    d = d.copy()

    d['sender_addr'] = d['sender'].apply(first_address)
    if verbose:
        print(f"build_features(): dropped {(d['sender_addr'] == '').sum()} rows with no valid sender address")
    d = d[d['sender_addr'] != ''].copy()

    d['subject_clean'] = d['subject'].apply(clean_text)
    d['body_clean']    = d['body'].apply(clean_text)
    empty = (d['subject_clean'] + d['body_clean']).str.len() == 0
    if verbose:
        print(f"build_features(): dropped {empty.sum()} rows with no text after cleaning")
    d = d[~empty].copy()

    # Address parts
    local              = d['sender_addr'].str.split('@').str[0]
    d['sender_domain'] = d['sender_addr'].str.split('@').str[1]
    d['sender_tld']    = d['sender_domain'].str.split('.').str[-1]
    raw_name           = d['sender'].apply(display_name)
    d['sender_name']   = raw_name.apply(norm_text)          # cleaned name, used for the brand checks
    receivers  = d['receiver'].apply(lambda s: [a.lower() for a in ADDR_RE.findall(s)])
    rcv_count  = receivers.apply(len)
    rcv_domain = receivers.apply(lambda a: a[0].split('@')[1] if a else 'none')

    # Address features
    d['sender_local_len_log'] = np.log1p(local.str.len())
    d['sender_local_digits']  = local.str.count(r'\d')
    d['sender_domain_len']    = d['sender_domain'].str.len()
    d['sender_subdomains']    = d['sender_domain'].str.count(r'\.')
    d['sender_domain_hyphen'] = d['sender_domain'].str.count('-')
    d['sender_domain_digits'] = d['sender_domain'].str.count(r'\d')
    d['sender_free_mail']     = d['sender_domain'].isin(FREE_MAIL).astype(int)
    d['sender_has_name']      = (d['sender_name'] != '').astype(int)
    d['receiver_count_log']   = np.log1p(rcv_count)
    d['receiver_missing']     = (rcv_count == 0).astype(int)
    d['same_domain']          = (d['sender_domain'] == rcv_domain).astype(int)
    d['sender_is_receiver']   = [int(a in r) for a, r in zip(d['sender_addr'], receivers)]

    # Text features (on raw text, where capitals and symbols still exist)
    full = d['subject'] + ' ' + d['body']
    txt  = d['subject_clean'] + ' ' + d['body_clean']
    d['subject_len_log']     = np.log1p(d['subject_clean'].str.len())
    d['body_len_log']        = np.log1p(d['body_clean'].str.len())
    d['exclaim_count_log']   = np.log1p(full.str.count('!'))
    d['money_symbols_log']   = np.log1p(full.str.count(r'[$£€]'))
    d['upper_ratio']         = d['body'].str.count(r'[A-Z]') / d['body'].str.len().clip(lower=1)
    d['subject_upper_ratio'] = d['subject'].str.count(r'[A-Z]') / d['subject'].str.len().clip(lower=1)
    d['subject_is_reply']    = d['subject'].str.contains(r'(?i)^\s*(?:re|fwd?)\s*:').astype(int)
    d['kw_urgency_log']      = np.log1p(txt.str.count(URGENCY_RE))
    d['kw_cred_log']         = np.log1p(txt.str.count(CRED_RE))

    # Unicode tricks (used as RULES, not ML features: the 2008 training data has almost none)
    name_subj         = raw_name + ' ' + d['subject']
    d['n_invisible']  = name_subj.str.count(INVIS)
    d['mixed_script'] = name_subj.apply(mixed_script)

    # Phishing indicators
    s_feats = pd.DataFrame([sender_risk(dom, nm) for dom, nm in zip(d['sender_domain'], d['sender_name'])],
                           index=d.index)
    u_feats = pd.DataFrame([url_risk(b) for b in d['body']], index=d.index)
    u_feats['u_count'] = np.log1p(u_feats['u_count'])
    return pd.concat([d, s_feats, u_feats], axis=1)

# Numeric ML features (address + text + phishing indicators)
NUM_FEATURES = ['sender_local_len_log', 'sender_local_digits', 'sender_domain_len', 'sender_subdomains',
                'sender_domain_hyphen', 'sender_domain_digits', 'sender_free_mail', 'sender_has_name',
                'receiver_count_log', 'receiver_missing', 'same_domain', 'sender_is_receiver',
                'subject_len_log', 'body_len_log', 'exclaim_count_log', 'money_symbols_log', 'upper_ratio',
                'subject_upper_ratio', 'subject_is_reply', 'kw_urgency_log', 'kw_cred_log',
                's_typosquat', 's_leet', 's_brand_in_domain', 's_brand_in_subdomain', 's_display_brand_mismatch',
                's_punycode', 's_nonascii', 's_susp_tld',
                'u_count', 'u_ip', 'u_shortener', 'u_at', 'u_anchor_mismatch']

# Strong deception indicators used as hard RULES in the hybrid system (ML OR rules)
STRONG = ['s_typosquat', 's_leet', 's_display_brand_mismatch', 's_brand_in_subdomain',
          'u_ip', 'u_at', 'u_anchor_mismatch', 'n_invisible', 'mixed_script']


# Deployment: score ONE email (used by the milter)

def numeric_matrix_p(d, prep):
    """Same as numeric_matrix(), but uses the SAVED feature lists, not the globals."""
    tl = prep['top_tlds']
    tld = pd.DataFrame({f'TLD_{t}': (d['sender_tld'] == t).astype(int) for t in tl}, index=d.index)
    tld['TLD_other'] = (~d['sender_tld'].isin(tl)).astype(int)
    return pd.concat([d[prep['num_features']], tld], axis=1)[prep['features']]

def featurize_one(sender, receiver, subject, body, prep):
    """One email -> (feature row, list of strong rules that fired). (None, None) = ML can't score it."""
    d = pd.DataFrame([{'sender': sender or '', 'receiver': receiver or '',
                       'subject': subject or '', 'body': body or ''}])
    if not first_address(d.at[0, 'sender']):
        return None, None                      # no From address -> rules + VirusTotal only
    f = build_features(d, verbose=False)
    if f.empty:
        return None, None                      # no text -> rules + VirusTotal only
    X = hstack([csr_matrix(prep['scaler'].transform(numeric_matrix_p(f, prep))),
                prep['text_tf'].transform(f)]).tocsr()
    fired = [r for r in prep['strong_rules'] if f.iloc[0][r] > 0]
    return X, fired