from typing import List, Sequence, Tuple, Optional, Dict, Union, Callable
import streamlit as st
import spacy
from spacy.language import Language
from spacy import displacy
import pandas as pd

from .util import load_model, process_text, get_html, get_svg, get_color_styles


# fmt: off
NER_ATTRS = ["text", "label_", "start", "end", "start_char", "end_char"]
TOKEN_ATTRS = ["idx", "text", "lemma_", "pos_", "tag_", "dep_", "head", "morph",
               "ent_type_", "ent_iob_", "shape_", "is_alpha", "is_ascii",
               "is_digit", "is_punct", "like_num", "is_sent_start"]
# fmt: on



def visualize(
    models: Union[List[str], Dict[str, str]],
    default_text: str = "",
    default_model: Optional[str] = None,
    visualizers: List[str] = ["parser", "ner", "textcat", "similarity", "tokens"],
    ner_labels: Optional[List[str]] = None,
    ner_attrs: List[str] = NER_ATTRS,
    similarity_texts: Tuple[str, str] = ("apple", "orange"),
    token_attrs: List[str] = TOKEN_ATTRS,
    show_json_doc: bool = True,
    show_meta: bool = True,
    show_config: bool = True,
    show_visualizer_select: bool = False,
    show_pipeline_info: bool = True,
    sidebar_title: Optional[str] = None,
    sidebar_description: Optional[str] = None,
    show_logo: bool = True,
    color: Optional[str] = "#09A3D5",
    key: Optional[str] = None,
    get_default_text: Callable[[Language], str] = None,
) -> None:
    """Встроить полный визуализатор с выбранными компонентами."""
    if color:
        st.write(get_color_styles(color), unsafe_allow_html=True)
    #if show_logo:
    #    st.sidebar.markdown(LOGO, unsafe_allow_html=True)
    if sidebar_title:
        st.sidebar.title(sidebar_title)
    if sidebar_description:
        st.sidebar.markdown(sidebar_description)

    # Allow both dict of model name / description as well as list of names
    model_names = models
    format_func = str
    if isinstance(models, dict):
        format_func = lambda name: models.get(name, name)
        model_names = list(models.keys())

    default_model_index = (
        model_names.index(default_model)
        if default_model is not None and default_model in model_names
        else 0
    )
    spacy_model = st.sidebar.selectbox(
        "Модель",
        model_names,
        index=default_model_index,
        key=f"{key}_visualize_models",
        format_func=format_func,
    )
    model_load_state = st.info(f"Загруженная модель '{spacy_model}'...")
    nlp = load_model(spacy_model)
    model_load_state.empty()

    if show_pipeline_info:
        st.sidebar.subheader("Информация о модели")
        desc = f"""<p style="font-size: 0.85em; line-height: 1.5"><strong>{spacy_model}:</strong> <code>v{nlp.meta['version']}</code>. {nlp.meta.get("description", "")}</p>"""
        st.sidebar.markdown(desc, unsafe_allow_html=True)

    if show_visualizer_select:
        active_visualizers = st.sidebar.multiselect(
            "Визуализаторы",
            options=visualizers,
            default=list(visualizers),
            key=f"{key}_viz_select",
        )
    else:
        active_visualizers = visualizers

    default_text = (
        get_default_text(nlp) if get_default_text is not None else default_text
    )
    text = st.text_area("Текст для анализа", default_text, key=f"{key}_visualize_text")
    doc = process_text(spacy_model, text)

    if "parser" in visualizers and "parser" in active_visualizers:
        visualize_parser(doc, key=key)
    if "ner" in visualizers and "ner" in active_visualizers:
        ner_labels = ner_labels or nlp.get_pipe("ner").labels
        visualize_ner(doc, labels=ner_labels, attrs=ner_attrs, key=key)
    if "textcat" in visualizers and "textcat" in active_visualizers:
        visualize_textcat(doc)
    if "similarity" in visualizers and "similarity" in active_visualizers:
        visualize_similarity(nlp, key=key)
    if "tokens" in visualizers and "tokens" in active_visualizers:
        visualize_tokens(doc, attrs=token_attrs, key=key)

    if show_json_doc or show_meta or show_config:
        st.header("Информация о представлении модели")
        if show_json_doc:
            json_doc_exp = st.beta_expander("JSON Doc")
            json_doc_exp.json(doc.to_json())

        if show_meta:
            meta_exp = st.beta_expander("Pipeline meta.json")
            meta_exp.json(nlp.meta)

        if show_config:
            config_exp = st.beta_expander("Pipeline config.cfg")
            config_exp.code(nlp.config.to_str())

    #st.sidebar.markdown(
    #    FOOTER,
    #    unsafe_allow_html=True,
    #)


def visualize_parser(
    doc: spacy.tokens.Doc,
    *,
    title: Optional[str] = "Теги синтаксического анализа зависимостей и частей речи",
    key: Optional[str] = None,
) -> None:
    """Визуализатор для анализа зависимостей."""
    if title:
        st.header(title)
    cols = st.beta_columns(4)
    split_sents = cols[0].checkbox(
        "Разделить предложения", value=True, key=f"{key}_parser_split_sents"
    )
    options = {
        "collapse_punct": cols[1].checkbox(
            "Свернуть пунктуацию", value=True, key=f"{key}_parser_collapse_punct"
        ),
        "collapse_phrases": cols[2].checkbox(
            "Свернуть фразы", key=f"{key}_parser_collapse_phrases"
        ),
        "compact": cols[3].checkbox("Компактный режим", key=f"{key}_parser_compact"),
    }
    docs = [span.as_doc() for span in doc.sents] if split_sents else [doc]
    for sent in docs:
        html = displacy.render(sent, options=options, style="dep")
        # Double newlines seem to mess with the rendering
        html = html.replace("\n\n", "\n")
        if split_sents and len(docs) > 1:
            st.markdown(f"> {sent.text}")
        st.write(get_svg(html), unsafe_allow_html=True)


def visualize_ner(
    doc: spacy.tokens.Doc,
    *,
    labels: Sequence[str] = tuple(),
    attrs: List[str] = NER_ATTRS,
    show_table: bool = True,
    title: Optional[str] = "Именованные сущности",
    colors: Dict[str, str] = {},
    key: Optional[str] = None,
) -> None:
    """Визуализатор именованных сущностей."""
    if title:
        st.header(title)
    exp = st.beta_expander("Выберите метки объекта")
    label_select = exp.multiselect(
        "Метки объектов",
        options=labels,
        default=list(labels),
        key=f"{key}_ner_label_select",
    )
    html = displacy.render(
        doc, style="ent", options={"ents": label_select, "colors": colors}
    )
    style = "<style>mark.entity { display: inline-block }</style>"
    st.write(f"{style}{get_html(html)}", unsafe_allow_html=True)
    if show_table:
        data = [
            [str(getattr(ent, attr)) for attr in attrs]
            for ent in doc.ents
            if ent.label_ in labels
        ]
        df = pd.DataFrame(data, columns=attrs)
        st.dataframe(df)


def visualize_textcat(
    doc: spacy.tokens.Doc, *, title: Optional[str] = "Классификация текста"
) -> None:
    """Визуализатор текстовых категорий."""
    if title:
        st.header(title)
    st.markdown(f"> {doc.text}")
    df = pd.DataFrame(doc.cats.items(), columns=("Label", "Score"))
    st.dataframe(df)


def visualize_similarity(
    nlp: spacy.language.Language,
    default_texts: Tuple[str, str] = ("яблоко", "красное"),
    *,
    threshold: float = 0.5,
    title: Optional[str] = "Векторы & Сходство",
    key: Optional[str] = None,
) -> None:
    """Визуализатор семантического сходства с использованием векторов слов."""
    meta = nlp.meta.get("vectors", {})
    if title:
        st.header(title)
    if not meta.get("width", 0):
        st.warning("В модели нет доступных векторов.")
    else:
        cols = st.beta_columns(2)
        text1 = cols[0].text_input(
            "Текст или слово 1", default_texts[0], key=f"{key}_similarity_text1"
        )
        text2 = cols[1].text_input(
            "Текст или слово 2", default_texts[1], key=f"{key}_similarity_text2"
        )
        doc1 = nlp.make_doc(text1)
        doc2 = nlp.make_doc(text2)
        similarity = doc1.similarity(doc2)
        similarity_text = f"**Вероятность:** `{similarity}`"
        if similarity > threshold:
            st.success(similarity_text)
        else:
            st.error(similarity_text)

        exp = st.beta_expander("Информация о векторном сходстве текста или слов")
        exp.code(meta)


def visualize_tokens(
    doc: spacy.tokens.Doc,
    *,
    attrs: List[str] = TOKEN_ATTRS,
    title: Optional[str] = "Атрибуты токенов",
    key: Optional[str] = None,
) -> None:
    """Визуализатор атрибутов токена."""
    if title:
        st.header(title)
    exp = st.beta_expander("Выбор атрибутов токена")
    selected = exp.multiselect(
        "Атрибуты токена",
        options=attrs,
        default=list(attrs),
        key=f"{key}_tokens_attr_select",
    )
    data = [[str(getattr(token, attr)) for attr in selected] for token in doc]
    df = pd.DataFrame(data, columns=selected)
    st.dataframe(df)
