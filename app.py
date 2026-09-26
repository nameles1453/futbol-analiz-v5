import os, math, requests
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Futbol Analiz PRO V6",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.block-container{padding:.8rem .7rem 2rem;max-width:1100px}
.stButton>button{width:100%;min-height:48px;border-radius:12px;font-weight:700}
.card{border:1px solid rgba(128,128,128,.25);border-radius:16px;padding:14px;margin:7px 0;background:rgba(128,128,128,.06)}
.big{font-size:1.35rem;font-weight:800}.muted{opacity:.72}
</style>
""", unsafe_allow_html=True)

BASE = "https://v3.football.api-sports.io"

def get_key():
    try:
        return st.secrets["APIFOOTBALL_KEY"]
    except Exception:
        return os.getenv("APIFOOTBALL_KEY", "")

KEY = get_key()

@st.cache_data(ttl=900, show_spinner=False)
def api(endpoint, params):
    if not KEY:
        raise RuntimeError("APIFOOTBALL_KEY bulunamadı.")
    r = requests.get(
        BASE + endpoint,
        headers={"x-apisports-key": KEY},
        params=params,
        timeout=30
    )
    r.raise_for_status()
    d = r.json()
    if d.get("errors"):
        raise RuntimeError(str(d["errors"]))
    return d.get("response", [])

def team_search(q):
    tr = str.maketrans({"ç":"c","Ç":"C","ğ":"g","Ğ":"G","ı":"i","İ":"I","ö":"o","Ö":"O","ş":"s","Ş":"S","ü":"u","Ü":"U"})
    return api("/teams", {"search": q.translate(tr).strip()})

def fixture_by_id(fid):
    rows = api("/fixtures", {"id": int(fid)})
    return rows[0] if rows else None

# Ücretsiz API planı 2026 sezonunu season parametresiyle istemeye izin vermeyebiliyor.
# Bu nedenle son tamamlanmış maçları "last" ile çekiyoruz ve hedef tarihten sonrasını
# modelden çıkarıyoruz. Böylece 2026 maçlarında "season required" hatasına takılmıyoruz.
def fixtures_before(tid, date, n=20):
    """
    Current/free API-Football plans can reject both season=2026 and last=...
    for team-history requests. Historical form is therefore optional here.
    We try a small date window only when available; otherwise return [] and
    let the model use neutral priors instead of crashing.
    """
    target = datetime.strptime(date, "%Y-%m-%d").date()
    found = []

    # Date-based fixture calls do not require the team/season/last parameters.
    # Keep the window small to avoid exhausting the free 100-request daily quota.
    for days_back in range(1, 8):
        day = target - timedelta(days=days_back)
        try:
            rows = api("/fixtures", {
                "date": day.isoformat(),
                "timezone": "Europe/Istanbul"
            })
            for x in rows:
                teams = x.get("teams", {})
                if (
                    teams.get("home", {}).get("id") == tid
                    or teams.get("away", {}).get("id") == tid
                ) and x.get("fixture", {}).get("status", {}).get("short") == "FT":
                    found.append(x)
        except Exception:
            continue

    return sorted(
        found,
        key=lambda x: x.get("fixture", {}).get("date", ""),
        reverse=True
    )[:n]

def h2h(h, a, date, n=10):
    try:
        rows = api("/fixtures/headtohead", {"h2h": f"{h}-{a}"})
    except Exception:
        return []
    cut = date + "T00:00:00+00:00"
    rows = [
        x for x in rows
        if x.get("fixture", {}).get("date", "") < cut
    ]
    return sorted(
        rows,
        key=lambda x: x.get("fixture", {}).get("date", ""),
        reverse=True
    )[:n]

def goals(f, tid):
    hg = f["goals"]["home"] or 0
    ag = f["goals"]["away"] or 0
    return (hg, ag) if f["teams"]["home"]["id"] == tid else (ag, hg)

def fhgoals(f, tid):
    x = f.get("score", {}).get("halftime", {}) or {}
    hg = x.get("home") or 0
    ag = x.get("away") or 0
    return (hg, ag) if f["teams"]["home"]["id"] == tid else (ag, hg)

def summary(rows, tid, venue=None):
    g = []
    for f in rows:
        if venue == "home" and f["teams"]["home"]["id"] != tid:
            continue
        if venue == "away" and f["teams"]["away"]["id"] != tid:
            continue
        gf, ga = goals(f, tid)
        hf, ha = fhgoals(f, tid)
        r = "W" if gf > ga else "D" if gf == ga else "L"
        g.append((gf, ga, hf, ha, r))

    if not g:
        return {
            "n": 0, "gf": 1, "ga": 1,
            "fhgf": .5, "fhga": .5,
            "ppg": 1, "w": 0, "d": 0, "l": 0
        }

    n = len(g)
    return {
        "n": n,
        "gf": sum(x[0] for x in g) / n,
        "ga": sum(x[1] for x in g) / n,
        "fhgf": sum(x[2] for x in g) / n,
        "fhga": sum(x[3] for x in g) / n,
        "ppg": sum(3 if x[4] == "W" else 1 if x[4] == "D" else 0 for x in g) / n,
        "w": sum(x[4] == "W" for x in g),
        "d": sum(x[4] == "D" for x in g),
        "l": sum(x[4] == "L" for x in g)
    }

def pmf(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)

def probs(lh, la, m=8):
    q = [
        [pmf(i, lh) * pmf(j, la) for j in range(m + 1)]
        for i in range(m + 1)
    ]

    one = sum(q[i][j] for i in range(m + 1) for j in range(m + 1) if i > j)
    draw = sum(q[i][j] for i in range(m + 1) for j in range(m + 1) if i == j)
    two = sum(q[i][j] for i in range(m + 1) for j in range(m + 1) if i < j)

    def over(n):
        return 1 - sum(
            q[i][j]
            for i in range(m + 1)
            for j in range(m + 1)
            if i + j <= n
        )

    btts = (
        1
        - sum(q[0][j] for j in range(m + 1))
        - sum(q[i][0] for i in range(m + 1))
        + q[0][0]
    )

    scores = sorted(
        [(q[i][j], f"{i}-{j}") for i in range(m + 1) for j in range(m + 1)],
        reverse=True
    )[:10]

    return {
        "1": one, "X": draw, "2": two,
        "O1.5": over(1), "O2.5": over(2), "O3.5": over(3),
        "U1.5": 1 - over(1), "U2.5": 1 - over(2), "U3.5": 1 - over(3),
        "BTTS": btts, "NO_BTTS": 1 - btts,
        "scores": scores
    }

def make_model(hid, aid, hr, ar, hh):
    ho = summary(hr, hid)
    aw = summary(ar, aid)
    hhome = summary(hr, hid, "home")
    aaway = summary(ar, aid, "away")
    hs = summary(hh, hid) if hh else {"gf": 1, "ga": 1}

    ha = .55 * ho["gf"] + .45 * hhome["gf"]
    hd = .55 * ho["ga"] + .45 * hhome["ga"]
    aa = .55 * aw["gf"] + .45 * aaway["gf"]
    ad = .55 * aw["ga"] + .45 * aaway["ga"]

    lh = max(.15, min(4.5, .52 * ha + .28 * ad + .20 * hs["gf"]))
    la = max(.15, min(4.2, .52 * aa + .28 * hd + .20 * hs["ga"]))

    fh = probs(
        max(.05, .55 * ho["fhgf"] + .45 * aw["fhga"]),
        max(.05, .55 * aw["fhgf"] + .45 * ho["fhga"]),
        5
    )

    return {
        "lh": lh,
        "la": la,
        "p": probs(lh, la),
        "fh": fh,
        "home": ho,
        "away": aw
    }

def get_odds(fid, p):
    rows = api("/odds", {"fixture": fid})
    vals = {}

    for row in rows:
        for bm in row.get("bookmakers", []):
            for bet in bm.get("bets", []):
                for v in bet.get("values", []):
                    try:
                        vals.setdefault(
                            (bet.get("name"), v.get("value")), []
                        ).append(float(v.get("odd")))
                    except Exception:
                        pass

    aliases = {
        "1": ("Match Winner", "Home"),
        "X": ("Match Winner", "Draw"),
        "2": ("Match Winner", "Away"),
        "O1.5": ("Goals Over/Under", "Over 1.5"),
        "U1.5": ("Goals Over/Under", "Under 1.5"),
        "O2.5": ("Goals Over/Under", "Over 2.5"),
        "U2.5": ("Goals Over/Under", "Under 2.5"),
        "O3.5": ("Goals Over/Under", "Over 3.5"),
        "U3.5": ("Goals Over/Under", "Under 3.5"),
        "BTTS": ("Both Teams Score", "Yes"),
        "NO_BTTS": ("Both Teams Score", "No")
    }

    out = []
    for k, (market, label) in aliases.items():
        if (market, label) in vals:
            odd = sorted(vals[(market, label)])[len(vals[(market, label)]) // 2]
            imp = 1 / odd
            out.append([
                k,
                round(p[k] * 100, 1),
                odd,
                round(imp * 100, 1),
                round((p[k] - imp) * 100, 1),
                round(1 / p[k], 2)
            ])

    if not out:
        return pd.DataFrame()

    return pd.DataFrame(
        out,
        columns=["Tahmin", "Model %", "Oran", "Piyasa %", "Model farkı", "Adil oran"]
    ).sort_values("Model farkı", ascending=False)

def player_data(fid):
    result = {}
    for label, ep in [
        ("injuries", "/injuries"),
        ("lineups", "/fixtures/lineups"),
        ("players", "/fixtures/players")
    ]:
        try:
            result[label] = api(ep, {"fixture": fid})
        except Exception:
            result[label] = []
    return result

@st.cache_data(ttl=60, show_spinner=False)
def live_fixtures(): return api("/fixtures", {"live":"all", "timezone":"Europe/Istanbul"})
@st.cache_data(ttl=30, show_spinner=False)
def live_events(fid): return api("/fixtures/events", {"fixture":int(fid)})
@st.cache_data(ttl=60, show_spinner=False)
def live_stats(fid): return api("/fixtures/statistics", {"fixture":int(fid)})
def live_status_minute(f):
    s=f.get("fixture",{}).get("status",{})
    try: minute=int(s.get("elapsed") or 0)
    except Exception: minute=0
    return minute,s.get("short","")
def event_summary(events):
    goals=[]; reds=[]
    for e in events:
        typ=e.get("type"); detail=(e.get("detail") or "").lower(); team=e.get("team",{}).get("name","?"); minute=e.get("time",{}).get("elapsed","?")
        if typ=="Goal": goals.append(f"{minute}' {team}")
        if typ=="Card" and "red" in detail: reds.append(f"{minute}' {team}")
    return goals,reds
def live_probability(pre_lh,pre_la,minute,hg,ag,red_home=0,red_away=0):
    remaining=max(0,90-min(max(minute,0),90)); frac=remaining/90
    ah=max(.25,pre_lh*(1+.20*min(red_away,2)-.20*min(red_home,2))); aa=max(.25,pre_la*(1+.20*min(red_home,2)-.20*min(red_away,2)))
    tail=probs(ah*frac,aa*frac,7); result={"1":0.,"X":0.,"2":0.}; scores=[]
    for prob,score in tail["scores"]:
        x,y=map(int,score.split("-")); fx,fy=hg+x,ag+y; key="1" if fx>fy else "X" if fx==fy else "2"; result[key]+=prob; scores.append((prob,f"{fx}-{fy}"))
    total=sum(result.values()) or 1; result={k:v/total for k,v in result.items()}; lam=(ah+aa)*frac
    return result,sorted(scores,reverse=True)[:8],1-math.exp(-lam),1-math.exp(-lam)*(1+lam)

st.title("⚽ Futbol Analiz PRO V6")
st.caption("Mobil web sürümü • Tahminler istatistiksel model çıktısıdır; garanti değildir.")

if not KEY:
    st.error("API-Football anahtarı bulunamadı. Streamlit Secrets içine APIFOOTBALL_KEY ekleyin.")
    st.stop()

@st.cache_data(ttl=300, show_spinner=False)
def fixtures_on_date(date):
    return api("/fixtures", {"date": date, "timezone": "Europe/Istanbul"})

def match_label(f):
    league = f.get("league", {}).get("name", "-")
    country = f.get("league", {}).get("country", "-")
    h = f.get("teams", {}).get("home", {}).get("name", "-")
    a = f.get("teams", {}).get("away", {}).get("name", "-")
    status = f.get("fixture", {}).get("status", {}).get("short", "")
    tm = f.get("fixture", {}).get("date", "")

    try:
        tm = datetime.fromisoformat(
            tm.replace("Z", "+00:00")
        ).astimezone().strftime("%H:%M")
    except Exception:
        tm = ""

    return f"{tm} • {h} - {a} • {league} ({country}) • {status}"

def is_national(f):
    league = f.get("league", {})
    name = (league.get("name") or "").lower()
    typ = (league.get("type") or "").lower()

    keywords = [
        "world cup", "euro", "nations league",
        "qualifiers", "qualification", "friendlies",
        "copa america", "africa cup", "asian cup",
        "gold cup", "concacaf"
    ]

    return (
        typ in {"national", "international"}
        or any(x in name for x in keywords)
    )

tab1, tab2, tab3, tab4 = st.tabs(["🎯 TAHMİN", "🔴 CANLI", "📈 BACKTEST", "📱 KURULUM"])

with tab1:
    st.subheader("📅 Bugünün / seçilen günün maçları")

    fixture_date = st.date_input(
        "Maç günü",
        datetime.now().date(),
        key="fixture_date"
    )

    try:
        today_fixtures = fixtures_on_date(fixture_date.isoformat())

        show_national = st.checkbox(
            "🌍 Millî takım / uluslararası maçları göster",
            True
        )
        show_club = st.checkbox(
            "🏟️ Kulüp maçlarını göster",
            True
        )

        pool = []
        for f in today_fixtures:
            nat = is_national(f)
            if (nat and show_national) or ((not nat) and show_club):
                pool.append(f)

        pool = sorted(
            pool,
            key=lambda f: f.get("fixture", {}).get("date", "")
        )

        if pool:
            labels = [match_label(f) for f in pool]
            selected = st.selectbox("Maç seç", labels, index=0)
            chosen = pool[labels.index(selected)]

            auto_h = chosen["teams"]["home"]["name"]
            auto_a = chosen["teams"]["away"]["name"]
            auto_fid = int(chosen["fixture"]["id"])
            auto_date = chosen["fixture"]["date"][:10]

            st.success(
                f"Seçilen maç: {auto_h} — {auto_a} | Fixture ID: {auto_fid}"
            )
        else:
            st.info(
                "Bu filtrelerle seçilen günde maç bulunamadı. "
                "Aşağıdan takımları elle girebilirsin."
            )
            auto_h = auto_a = ""
            auto_fid = 0
            auto_date = fixture_date.isoformat()

    except Exception as e:
        st.warning(f"Maç listesi alınamadı: {e}")
        auto_h = auto_a = ""
        auto_fid = 0
        auto_date = fixture_date.isoformat()

    st.subheader("✍️ Elle maç girişi")

    c1, c2 = st.columns(2)

    with c1:
        hn = st.text_input(
            "Ev sahibi",
            value=auto_h,
            placeholder="Galatasaray"
        )

    with c2:
        an = st.text_input(
            "Deplasman",
            value=auto_a,
            placeholder="Fenerbahçe"
        )

    date = st.date_input(
        "Maç tarihi",
        datetime.fromisoformat(auto_date).date(),
        key="analysis_date"
    )

    fid = st.number_input(
        "Fixture ID (oran + oyuncu verileri için)",
        0,
        99999999,
        auto_fid
    )

    if st.button("⚡ MAÇI TAHMİN ET", type="primary"):
        try:
            selected_fixture = None
            if int(fid) > 0:
                try:
                    selected_fixture = fixture_by_id(int(fid))
                except Exception:
                    selected_fixture = None

            if selected_fixture:
                h = selected_fixture["teams"]["home"]
                a = selected_fixture["teams"]["away"]
            else:
                hs = team_search(hn)
                ats = team_search(an)
                if not hs or not ats:
                    raise RuntimeError("Takım bulunamadı.")
                h = hs[0]["team"]
                a = ats[0]["team"]

            hr = fixtures_before(h["id"], date.isoformat())
            ar = fixtures_before(a["id"], date.isoformat())
            hh = h2h(h["id"], a["id"], date.isoformat())

            m = make_model(
                h["id"],
                a["id"],
                hr,
                ar,
                hh
            )

            if not hr or not ar:
                st.warning(
                    "⚠️ API-Football ücretsiz paketinde 2026 takım geçmişi "
                    "kısıtlı olduğu için bazı form verileri kullanılamadı. "
                    "Tahmin mevcut erişilebilir veriler ve model öncülleri "
                    "üzerinden hesaplandı."
                )

            p = m["p"]
            fh = m["fh"]

            st.success(f"{h['name']} — {a['name']}")

            r = sorted(
                [("1", p["1"]), ("X", p["X"]), ("2", p["2"])],
                key=lambda x: x[1],
                reverse=True
            )

            st.subheader("🎯 Maç sonucu tahmini")

            cols = st.columns(3)
            for c, (k, v) in zip(cols, r):
                c.metric(k, f"{v * 100:.1f}%")

            st.markdown(
                f'<div class="card">'
                f'<div class="big">Model tahmini: {r[0][0]}</div>'
                f'<div class="muted">En yüksek model olasılığı: {r[0][1] * 100:.1f}%</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.metric(
                "Beklenen gol",
                f"{m['lh']:.2f} — {m['la']:.2f}"
            )

            st.subheader("⏱️ İlk yarı tahmini")

            fr = sorted(
                [("İY 1", fh["1"]), ("İY X", fh["X"]), ("İY 2", fh["2"])],
                key=lambda x: x[1],
                reverse=True
            )

            st.dataframe(
                pd.DataFrame(
                    [
                        ["İY 1", fh["1"] * 100],
                        ["İY X", fh["X"] * 100],
                        ["İY 2", fh["2"] * 100]
                    ],
                    columns=["Tahmin", "Olasılık %"]
                ),
                hide_index=True,
                use_container_width=True
            )

            st.markdown(
                f'<div class="card"><b>Modelin ilk yarı tahmini:</b> '
                f'{fr[0][0]} — {fr[0][1] * 100:.1f}%</div>',
                unsafe_allow_html=True
            )

            st.subheader("⚽ Alt / Üst ve KG")

            market = [
                ("Üst 1.5", p["O1.5"]),
                ("Alt 1.5", p["U1.5"]),
                ("Üst 2.5", p["O2.5"]),
                ("Alt 2.5", p["U2.5"]),
                ("Üst 3.5", p["O3.5"]),
                ("Alt 3.5", p["U3.5"]),
                ("KG Var", p["BTTS"]),
                ("KG Yok", p["NO_BTTS"])
            ]

            st.dataframe(
                pd.DataFrame(
                    [[x, round(v * 100, 1)] for x, v in market],
                    columns=["Tahmin", "Olasılık %"]
                ),
                hide_index=True,
                use_container_width=True
            )

            bm = max(market, key=lambda x: x[1])

            st.markdown(
                f'<div class="card">'
                f'<div class="big">En yüksek yan market: {bm[0]}</div>'
                f'<div class="muted">{bm[1] * 100:.1f}%</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.subheader("🔢 Skor tahmini")

            st.dataframe(
                pd.DataFrame(
                    [
                        {"Skor": s, "Olasılık %": round(v * 100, 2)}
                        for v, s in p["scores"]
                    ]
                ),
                hide_index=True,
                use_container_width=True
            )

            st.markdown(
                f'<div class="card"><b>En olası skor:</b> '
                f'{p["scores"][0][1]} — {p["scores"][0][0] * 100:.1f}%</div>',
                unsafe_allow_html=True
            )

            st.subheader("📊 Form")

            st.dataframe(
                pd.DataFrame(
                    [
                        [
                            "Ev sahibi",
                            m["home"]["n"],
                            m["home"]["w"],
                            m["home"]["d"],
                            m["home"]["l"],
                            round(m["home"]["gf"], 2),
                            round(m["home"]["ga"], 2)
                        ],
                        [
                            "Deplasman",
                            m["away"]["n"],
                            m["away"]["w"],
                            m["away"]["d"],
                            m["away"]["l"],
                            round(m["away"]["gf"], 2),
                            round(m["away"]["ga"], 2)
                        ]
                    ],
                    columns=["Takım", "Maç", "G", "B", "M", "Gol", "Yenen"]
                ),
                hide_index=True,
                use_container_width=True
            )

            if fid:
                st.subheader("💰 Oran ↔ model")

                try:
                    od = get_odds(int(fid), p)
                    if not od.empty:
                        st.dataframe(
                            od,
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.info("Bu Fixture ID için oran verisi bulunamadı.")
                except Exception as e:
                    st.info(f"Oran verisi alınamadı: {e}")

                pc = player_data(int(fid))

                st.subheader("👥 Oyuncu / sakatlık / ilk 11")

                st.write(
                    f"Sakatlık: {len(pc['injuries'])} • "
                    f"İlk 11: {len(pc['lineups'])} • "
                    f"Oyuncu: {len(pc['players'])}"
                )

        except Exception as e:
            st.error(str(e))

with tab2:
    st.subheader("🔴 CANLI MAÇ ANALİZİ")
    st.caption("Canlı veriler yaklaşık 30–60 saniyelik yenileme ile alınır. Olasılıklar model tahminidir; garanti değildir.")
    if st.button("🔄 CANLI MAÇLARI YENİLE", key="refresh_live"): st.cache_data.clear()
    try:
        lf=live_fixtures()
        if not lf:
            st.info("Şu anda API'de canlı maç görünmüyor.")
        else:
            labels=[match_label(f) for f in lf]; choice=st.selectbox("Canlı maç seç",labels,key="live_match"); f=lf[labels.index(choice)]
            fid=int(f["fixture"]["id"]); ht=f["teams"]["home"]; at=f["teams"]["away"]; hg=int(f["goals"]["home"] or 0); ag=int(f["goals"]["away"] or 0); minute,status=live_status_minute(f)
            st.success(f"{ht['name']} {hg} - {ag} {at['name']} • {minute}' • Fixture ID: {fid}")
            try: events=live_events(fid)
            except Exception: events=[]
            goal_events,red_events=event_summary(events); c1,c2,c3=st.columns(3); c1.metric("Dakika",f"{minute}'"); c2.metric("Gol",str(hg+ag)); c3.metric("Kırmızı",str(len(red_events)))
            if goal_events: st.write("⚽", " • ".join(goal_events))
            if red_events: st.write("🟥", " • ".join(red_events))
            red_home=sum(1 for e in events if e.get("type")=="Card" and "red" in (e.get("detail") or "").lower() and e.get("team",{}).get("id")==ht.get("id")); red_away=sum(1 for e in events if e.get("type")=="Card" and "red" in (e.get("detail") or "").lower() and e.get("team",{}).get("id")==at.get("id"))
            try:
                d=f["fixture"]["date"][:10]; m=make_model(ht["id"],at["id"],fixtures_before(ht["id"],d),fixtures_before(at["id"],d),h2h(ht["id"],at["id"],d)); lp,scores,p1,p2=live_probability(m["lh"],m["la"],minute,hg,ag,red_home,red_away)
                st.subheader("🎯 Canlı sonuç olasılığı"); cc=st.columns(3); cc[0].metric("1",f"{lp['1']*100:.1f}%"); cc[1].metric("X",f"{lp['X']*100:.1f}%"); cc[2].metric("2",f"{lp['2']*100:.1f}%")
                top=max(lp.items(),key=lambda x:x[1]); st.markdown(f'<div class="card"><div class="big">Canlı model: {top[0]} — {top[1]*100:.1f}%</div><div class="muted">Skor + dakika + kırmızı kartlar + maç öncesi gol beklentisi kullanıldı.</div></div>',unsafe_allow_html=True)
                st.subheader("⚽ Kalan sürede gol"); c1,c2=st.columns(2); c1.metric("En az 1 gol",f"{p1*100:.1f}%"); c2.metric("2+ gol",f"{p2*100:.1f}%")
                st.subheader("🔢 Tahmini maç sonu skorları"); st.dataframe(pd.DataFrame([{"Skor":s,"Olasılık %":round(v*100,1)} for v,s in scores]),hide_index=True,use_container_width=True)
                st.subheader("📊 Canlı istatistikler")
                try:
                    stats=live_stats(fid); rows=[]
                    for side in stats:
                        tn=side.get("team",{}).get("name","?")
                        for item in side.get("statistics",[]): rows.append([item.get("type",""),tn,item.get("value","")])
                    if rows: st.dataframe(pd.DataFrame(rows,columns=["İstatistik","Takım","Değer"]),hide_index=True,use_container_width=True)
                    else: st.info("Bu maç için canlı istatistik verisi yok.")
                except Exception as e: st.info(f"Canlı istatistik alınamadı: {e}")
                st.caption("⚠️ Bu yaklaşık bir in-play modeldir; bookmaker canlı oranı veya garanti değildir.")
            except Exception as e: st.error(f"Canlı model hesaplanamadı: {e}")
    except Exception as e: st.error(f"Canlı maç verisi alınamadı: {e}")

with tab3:
    st.subheader("Walk-forward backtest")

    st.info(
        "Ücretsiz API-Football planında 2026 sezonu geçmiş-sezon "
        "sorgularında kısıtlı olabilir. Bu bölüm için erişiminiz olan "
        "sezonu seçin."
    )

    lid = st.number_input("Lig ID", 1, 99999, 203)
    season = st.number_input(
        "Sezon",
        2000,
        2035,
        2024
    )
    n = st.number_input("Test maçı", 5, 50, 15)
    days = st.slider("Geçmiş dönem", 30, 365, 180)

    if st.button("BACKTEST ÇALIŞTIR", type="primary"):
        try:
            end = datetime.now().date()
            start = end - timedelta(days=days)

            fs = api(
                "/fixtures",
                {
                    "league": lid,
                    "season": season,
                    "from": start.isoformat(),
                    "to": end.isoformat(),
                    "status": "FT"
                }
            )

            fs = sorted(
                fs,
                key=lambda x: x["fixture"]["date"]
            )[-int(n):]

            rows = []

            for f in fs:
                d = f["fixture"]["date"][:10]
                hid = f["teams"]["home"]["id"]
                aid = f["teams"]["away"]["id"]

                try:
                    m = make_model(
                        hid,
                        aid,
                        fixtures_before(hid, d),
                        fixtures_before(aid, d),
                        h2h(hid, aid, d)
                    )

                    p = m["p"]

                    hg = f["goals"]["home"] or 0
                    ag = f["goals"]["away"] or 0

                    actual = (
                        "1" if hg > ag
                        else "X" if hg == ag
                        else "2"
                    )

                    pred = max(
                        [
                            ("1", p["1"]),
                            ("X", p["X"]),
                            ("2", p["2"])
                        ],
                        key=lambda x: x[1]
                    )

                    rows.append(
                        [
                            d,
                            f["teams"]["home"]["name"],
                            f["teams"]["away"]["name"],
                            actual,
                            pred[0],
                            round(pred[1] * 100, 1),
                            actual == pred[0]
                        ]
                    )

                except Exception:
                    pass

            if rows:
                df = pd.DataFrame(
                    rows,
                    columns=[
                        "Tarih", "Ev", "Dep", "Gerçek",
                        "Tahmin", "Model %", "İsabet"
                    ]
                )

                st.metric(
                    "1X2 isabet",
                    f"{df['İsabet'].mean() * 100:.1f}%"
                )

                st.dataframe(
                    df,
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.warning("Veri alınamadı.")

        except Exception as e:
            st.error(str(e))

with tab4:
    st.subheader("📱 Telefonda kullanım")
    st.write(
        "GitHub'a app.py ve requirements.txt yükle; "
        "Streamlit Community Cloud'da deploy et; "
        "Secrets'a APIFOOTBALL_KEY ekle; "
        "oluşan streamlit.app adresini telefondan aç."
    )
    st.info(
        "Chrome/Safari üzerinden ana ekrana ekleyerek uygulama gibi kullanabilirsin."
    )
