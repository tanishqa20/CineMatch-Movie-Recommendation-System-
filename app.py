
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO
from html import escape
import ast
from remotezip import RemoteZip
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# -------------------- PAGE + FILES --------------------

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MOVIES_FILE = DATA_DIR / "clean_movies.csv"
RATINGS_FILE = DATA_DIR / "clean_ratings.csv"
LINKS_FILE = DATA_DIR / "links.csv"

POSTER_ZIP = (
    "https://huggingface.co/datasets/Tanishqa20/"
    "cinematch-posters/resolve/main/"
    "selected_movie_posters_small.zip"
)

# -------------------- CSS --------------------

st.markdown("""
<style>
#MainMenu, footer, header {
    visibility: hidden;
}

.stApp {
    background: #070b0f;
    color: #f4f5f7;
    font-family: Arial, sans-serif;
}

.block-container {
    max-width: 1500px;
    padding: 0.5rem 2rem 1.3rem 2rem;
}

div[data-testid="stAppViewContainer"] {
    background: radial-gradient(
        ellipse at top,
        #10151c 0%,
        #070b0f 72%
    );
}

.top-brand {
    font-size: 19px;
    font-weight: 800;
    margin: 0 0 8px 14px;
    color: #f7f7f7 !important;
}

.top-brand span,
.hero h1 span {
    color: #f51b35 !important;
}

.hero {
    position: relative;
    overflow: hidden;
    border: 1px solid #4b252b;
    border-radius: 8px;
    min-height: 150px;
    padding: 23px 25px;
    margin-bottom: 18px;
    background:
        radial-gradient(
            ellipse at 77% 48%,
            rgba(142,0,21,.28),
            transparent 40%
        ),
        linear-gradient(
            100deg,
            #171114 0%,
            #100d11 58%,
            #25080e 100%
        );
}

.hero h1 {
    font-size: clamp(30px, 4vw, 40px);
    line-height: 1.1;
    margin: 0 0 10px 0;
    font-weight: 900;
    letter-spacing: -1px;
    color: #ffffff !important;
}

.hero p {
    font-size: 13px;
    line-height: 1.55;
    color: #e0dce1 !important;
    margin: 0;
    max-width: 520px;
}

.control-panel {
    background: linear-gradient(120deg, #0e141b, #0b1016);
    border: 1px solid #263341;
    border-radius: 9px;
    padding: 11px 14px 5px 14px;
    margin: 0 0 20px 0;
}

div[data-testid="stSelectbox"] label {
    color: #f0f2f5 !important;
    font-size: 12px !important;
}

/* FIX 1: Keep dropdown text and selected movie readable */

div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: #1b222c !important;
    border: 1px solid #2a3542 !important;
    border-radius: 7px;
    min-height: 39px;
}

div[data-testid="stSelectbox"] div[data-baseweb="select"] span,
div[data-testid="stSelectbox"] div[data-baseweb="select"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"] div {
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
}

div[data-testid="stSelectbox"] div[data-baseweb="select"] svg {
    fill: #f5f5f5 !important;
    color: #f5f5f5 !important;
}

div[data-baseweb="popover"],
div[data-baseweb="menu"],
ul[role="listbox"] {
    background-color: #151b24 !important;
}

li[role="option"],
li[role="option"] div,
li[role="option"] span {
    background-color: #151b24 !important;
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
}

li[role="option"]:hover {
    background-color: #303a49 !important;
}

div[data-testid="stButton"] > button {
    margin-top: 27px;
    width: 100%;
    min-height: 39px;
    background: #f20d2c;
    color: #ffffff !important;
    border: 0;
    border-radius: 7px;
    font-weight: 800;
    font-size: 12px;
    transition: .15s ease-in-out;
}

div[data-testid="stButton"] > button:hover {
    background: #c90b25;
    color: #ffffff !important;
    border: 0;
    transform: translateY(-1px);
}

.results-heading {
    border-left: 4px solid #f51b35;
    padding: 7px 0 7px 13px;
    margin: 8px 0 12px 0;
    color: #ffffff !important;
}

.results-heading h2 {
    color: #ffffff !important;
    font-size: 21px;
    margin: 0;
    font-weight: 850;
}

.result-subtitle {
    font-size: 12px;
    color: #d0d5de !important;
    margin: 0 0 13px 17px;
}

.selected-title {
    color: #ffffff !important;
    font-size: 28px;
    font-weight: 800;
    margin: 2px 0 15px 0;
}

.selected-meta {
    color: #e0e5ec !important;
    font-size: 14px;
    line-height: 1.9;
}

.selected-overview {
    color: #d0d6df !important;
    font-size: 14px;
    line-height: 1.8;
    margin-top: 14px;
}

.poster-placeholder {
    width: 100%;
    aspect-ratio: 2 / 3;
    background: linear-gradient(145deg, #1c2934, #10151c);
    color: #e0e5ec !important;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    font-size: 15px;
    border-radius: 7px;
}

.movie-card-title {
    color: #ffffff !important;
    font-size: 13px;
    font-weight: 800;
    margin-top: 8px;
    overflow-wrap: anywhere;
}

.movie-card-meta {
    color: #cbd2dc !important;
    font-size: 11px;
    line-height: 1.6;
    margin-top: 5px;
}

.star {
    color: #ffc83d !important;
    font-size: 14px;
}

div[data-testid="stAlert"] {
    background: #161b22;
    color: #ffffff !important;
}

div[data-testid="stAlert"] p {
    color: #ffffff !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #273442;
    background: #0c1117;
    border-radius: 8px;
}

@media (max-width: 800px) {
    .block-container {
        padding: .5rem .7rem 1rem .7rem;
    }

    .hero {
        padding: 18px;
    }

    .selected-title {
        font-size: 22px;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="top-brand">🎬 Cine<span>Match</span></div>',
    unsafe_allow_html=True,
)

st.markdown("""
<div class="hero">
    <h1>🎬 Cine<span>Match</span></h1>
    <p>
        Your next favourite movie is just one recommendation away.<br>
        Pick a film you love and discover similar ones based on<br>
        genres and plot descriptions.
    </p>
</div>
""", unsafe_allow_html=True)


# -------------------- FIX 2: CLEAN GENRE NAMES --------------------

def clean_genres(value):
    if pd.isna(value) or not str(value).strip():
        return "Not available"

    value = str(value).strip()

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            names = [
                str(item["name"])
                for item in parsed
                if isinstance(item, dict) and item.get("name")
            ]

            if names:
                return ", ".join(names)

    except (ValueError, SyntaxError, TypeError):
        pass

    return value


# -------------------- LOAD DATA --------------------

@st.cache_data(show_spinner=False)
def load_data():
    movies = pd.read_csv(MOVIES_FILE)
    ratings = pd.read_csv(RATINGS_FILE)
    links = pd.read_csv(LINKS_FILE)

    for col in ["tmdbId", "vote_average", "vote_count"]:
        if col in movies:
            movies[col] = pd.to_numeric(
                movies[col], errors="coerce"
            )

    for col in ["movieId", "userId", "rating"]:
        if col in ratings:
            ratings[col] = pd.to_numeric(
                ratings[col], errors="coerce"
            )

    for col in ["movieId", "tmdbId", "imdbId"]:
        if col in links:
            links[col] = pd.to_numeric(
                links[col], errors="coerce"
            )

    movies = movies.dropna(
        subset=["title", "tmdbId"]
    ).copy()

    movies["title"] = movies["title"].astype(str)

    for col in ["genres", "overview", "release_date"]:
        if col not in movies:
            movies[col] = ""

        movies[col] = movies[col].fillna("").astype(str)

    # Convert raw genre dictionaries into readable names.
    movies["genres"] = movies["genres"].apply(clean_genres)

    for col in ["vote_average", "vote_count"]:
        if col not in movies:
            movies[col] = 0

        movies[col] = pd.to_numeric(
            movies[col], errors="coerce"
        ).fillna(0)

    links = links.dropna(
        subset=["tmdbId"]
    ).drop_duplicates("tmdbId")

    movies = movies.merge(
        links[["tmdbId", "movieId", "imdbId"]],
        on="tmdbId",
        how="left",
        suffixes=("", "_link"),
    )

    if "movieId" not in movies:
        movies["movieId"] = np.nan

    if "imdbId" not in movies:
        movies["imdbId"] = np.nan

    year = movies["release_date"].str[:4]
    year = year.where(year.str.fullmatch(r"\d{4}"), "—")

    movies["display_title"] = (
        movies["title"] + " (" + year + ")"
    )

    duplicate = movies["display_title"].duplicated(keep=False)

    movies.loc[duplicate, "display_title"] += (
        " · "
        + movies.loc[duplicate, "tmdbId"].astype(int).astype(str)
    )

    ratings = ratings.dropna(
        subset=["userId", "movieId", "rating"]
    )

    return (
        movies.reset_index(drop=True),
        ratings,
        links.reset_index(drop=True),
    )


# -------------------- RECOMMENDATION MODELS --------------------

@st.cache_resource(show_spinner="Preparing recommendations...")
def build_models(_movies, _ratings):
    movies = _movies.copy()
    ratings = _ratings.copy()

    text_data = (
        movies["genres"] + " " + movies["overview"]
    ).tolist()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
    )

    content_matrix = vectorizer.fit_transform(text_data)

    valid = ratings[
        ratings["movieId"].isin(
            movies["movieId"].dropna()
        )
    ].copy()

    movie_ids = valid["movieId"].unique()

    movie_to_row = {
        int(mid): i for i, mid in enumerate(movie_ids)
    }

    user_ids = valid["userId"].unique()

    user_to_col = {
        uid: i for i, uid in enumerate(user_ids)
    }

    if len(movie_ids) and len(user_ids):
        rr = valid["movieId"].map(movie_to_row).to_numpy()
        cc = valid["userId"].map(user_to_col).to_numpy()
        vv = valid["rating"].to_numpy(dtype=float)

        rating_matrix = csr_matrix(
            (vv, (rr, cc)),
            shape=(len(movie_ids), len(user_ids)),
        )

        knn = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )

        knn.fit(rating_matrix)

    else:
        rating_matrix = csr_matrix((0, 0))
        knn = None

    return (
        content_matrix,
        movie_to_row,
        rating_matrix,
        knn,
        movie_ids,
    )


# -------------------- RECOMMENDATIONS --------------------

def recommend(
    selected_idx,
    method,
    count,
    movies,
    content_matrix,
    movie_to_row,
    rating_matrix,
    knn,
    movie_ids,
):
    content_scores = cosine_similarity(
        content_matrix[selected_idx],
        content_matrix,
    ).ravel()

    content_scores[selected_idx] = -1

    selected_movie_id = movies.iloc[selected_idx]["movieId"]

    collaborative_scores = np.zeros(
        len(movies),
        dtype=float,
    )

    if (
        pd.notna(selected_movie_id)
        and int(selected_movie_id) in movie_to_row
        and knn is not None
    ):
        row_idx = movie_to_row[int(selected_movie_id)]
        n = min(40, rating_matrix.shape[0])

        distances, neighbors = knn.kneighbors(
            rating_matrix[row_idx],
            n_neighbors=n,
        )

        movieid_to_indices = {}

        for i, mid in enumerate(movies["movieId"]):
            if pd.notna(mid):
                movieid_to_indices.setdefault(
                    int(mid), []
                ).append(i)

        for neighbor_idx, distance in zip(
            neighbors[0], distances[0]
        ):
            mid = int(movie_ids[neighbor_idx])

            for movie_idx in movieid_to_indices.get(mid, []):
                collaborative_scores[movie_idx] = max(
                    collaborative_scores[movie_idx],
                    1 - float(distance),
                )

    collaborative_scores[selected_idx] = -1

    if method == "Content-Based":
        scores = content_scores

    elif method == "Similarity-Based":
        scores = np.where(
            collaborative_scores > 0,
            collaborative_scores,
            content_scores * 0.85,
        )

    else:
        def norm(arr):
            arr = np.nan_to_num(arr, nan=0.0)
            lo, hi = arr.min(), arr.max()

            if hi > lo:
                return (arr - lo) / (hi - lo)

            return np.zeros_like(arr)

        popularity = movies["vote_average"].to_numpy(
            dtype=float
        )

        scores = (
            0.55 * norm(content_scores)
            + 0.35 * norm(collaborative_scores)
            + 0.10 * norm(popularity)
        )

    scores[selected_idx] = -1

    order = np.argsort(scores)[::-1]

    order = [
        i for i in order
        if i != selected_idx and scores[i] > 0
    ][:count]

    return movies.iloc[order].copy()


# -------------------- POSTER LOADING --------------------

@st.cache_resource(show_spinner=False)
def get_archive():
    archive = RemoteZip(POSTER_ZIP)
    filenames = set(archive.namelist())
    return archive, filenames


@st.cache_data(show_spinner=False, ttl=86400)
def get_poster(imdb_id):
    if pd.isna(imdb_id):
        return None

    try:
        filename = f"covers/{int(imdb_id):07d}.jpg"
        archive, filenames = get_archive()

        if filename in filenames:
            return archive.read(filename)

    except Exception:
        return None

    return None


def show_poster(movie):
    poster = get_poster(movie.get("imdbId"))

    if poster:
        st.image(
            BytesIO(poster),
            use_container_width=True,
        )
    else:
        st.markdown(
            '<div class="poster-placeholder">'
            '🎬<br>Poster unavailable</div>',
            unsafe_allow_html=True,
        )


def show_movie_card(movie):
    show_poster(movie)

    title = escape(str(movie.get("title", "Unknown title")))

    rating = pd.to_numeric(
        movie.get("vote_average", 0),
        errors="coerce",
    )

    if pd.isna(rating):
        rating = 0

    genres = escape(
        str(movie.get("genres", "") or "Genre unavailable")
    )

    st.markdown(
        f'<div class="movie-card-title">{title}</div>'
        f'<div class="movie-card-meta">'
        f'<span class="star">★</span> Rating: {rating:.1f}'
        f'<br>{genres}'
        f'</div>',
        unsafe_allow_html=True,
    )


# -------------------- CHECK FILES + PREPARE MODELS --------------------

try:
    if not all(
        p.exists()
        for p in [MOVIES_FILE, RATINGS_FILE, LINKS_FILE]
    ):
        st.error(
            "Please make sure clean_movies.csv, "
            "clean_ratings.csv, and links.csv "
            "are inside the data folder."
        )
        st.stop()

    movies, ratings, links = load_data()

    (
        content_matrix,
        movie_to_row,
        rating_matrix,
        knn,
        movie_ids,
    ) = build_models(movies, ratings)

except Exception as e:
    st.error(
        "CineMatch couldn't load its data. "
        "Check the CSV files and installed packages."
    )
    st.exception(e)
    st.stop()


# -------------------- HORIZONTAL CONTROLS --------------------

with st.container():
    st.markdown(
        '<div class="control-panel">',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(
        [2.1, 1.2, 1.45, 1.0],
        gap="small",
    )

    movie_options = movies["display_title"].tolist()

    default_index = next(
        (
            i for i, title in enumerate(movie_options)
            if title.lower().startswith("inception (")
        ),
        0,
    )

    with c1:
        selected_display = st.selectbox(
            "Search or select a movie",
            movie_options,
            index=default_index,
        )

    with c2:
        number = st.selectbox(
            "Number of recommendations",
            [5, 10, 15, 20],
            index=0,
        )

    with c3:
        method_label = st.selectbox(
            "Recommendation method",
            [
                "Similarity-Based",
                "Content-Based",
                "Hybrid",
            ],
            index=0,
        )

    with c4:
        find_clicked = st.button(
            "✨ Find My Movies",
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------- UPDATE RESULTS --------------------

selected_idx = int(
    movies.index[
        movies["display_title"] == selected_display
    ][0]
)

current_settings = (
    selected_display,
    method_label,
    number,
)

if (
    find_clicked
    or "last_settings" not in st.session_state
    or st.session_state["last_settings"] != current_settings
):
    st.session_state["results"] = recommend(
        selected_idx,
        method_label,
        number,
        movies,
        content_matrix,
        movie_to_row,
        rating_matrix,
        knn,
        movie_ids,
    )

    st.session_state["last_settings"] = current_settings


# -------------------- SELECTED MOVIE --------------------

selected_movie = movies.iloc[selected_idx]

st.markdown(
    '<div class="results-heading">'
    '<h2>🎬 Your Selected Movie</h2>'
    '</div>',
    unsafe_allow_html=True,
)

poster_col, info_col = st.columns(
    [1, 3],
    gap="medium",
)

with poster_col:
    show_poster(selected_movie)

with info_col:
    selected_title = escape(
        str(selected_movie.get("title", "Unknown title"))
    )

    selected_genres = escape(
        str(selected_movie.get("genres", "") or "Not available")
    )

    selected_date = escape(
        str(selected_movie.get("release_date", "") or "Not available")
    )

    selected_overview = escape(
        str(
            selected_movie.get("overview", "")
            or "No description available."
        )
    )

    selected_rating = pd.to_numeric(
        selected_movie.get("vote_average", 0),
        errors="coerce",
    )

    if pd.isna(selected_rating):
        selected_rating = 0

    st.markdown(
        f'<div class="selected-title">{selected_title}</div>'
        f'<div class="selected-meta">'
        f'<b>Genres:</b> {selected_genres}'
        f'<br><b>TMDB Rating:</b> {selected_rating:.1f} / 10'
        f'<br><b>Release Date:</b> {selected_date}'
        f'</div>'
        f'<div class="selected-overview">'
        f'{selected_overview}'
        f'</div>',
        unsafe_allow_html=True,
    )


# -------------------- RECOMMENDED MOVIES --------------------

st.markdown("---")

st.markdown(
    '<div class="results-heading">'
    '<h2>🍿 Recommended for you</h2>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="result-subtitle">'
    f'Based on your selection: '
    f'<b>{escape(selected_display)}</b>'
    f' &nbsp;·&nbsp; Method: '
    f'<b>{escape(method_label)}</b>'
    f'</div>',
    unsafe_allow_html=True,
)

results = st.session_state.get(
    "results",
    pd.DataFrame(),
)

if results.empty:
    st.info(
        "No recommendations found for this movie. "
        "Try another selection."
    )

else:
    columns_per_row = 5

    for start in range(0, len(results), columns_per_row):
        row = results.iloc[
            start:start + columns_per_row
        ]

        cols = st.columns(
            columns_per_row,
            gap="small",
        )

        for col, (_, movie) in zip(cols, row.iterrows()):
            with col:
                with st.container(border=True):
                    show_movie_card(movie)


# -------------------- FOOTER --------------------

st.markdown(
    '<div style="text-align:center;color:#929ba6;'
    'font-size:10px;margin-top:20px;">'
    'CineMatch · Built with Streamlit'
    '</div>',
    unsafe_allow_html=True,
)
