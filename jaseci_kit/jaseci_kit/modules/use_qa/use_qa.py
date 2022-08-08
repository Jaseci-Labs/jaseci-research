import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text  # noqa
from jaseci.actions.live_actions import jaseci_action
from typing import Union
from jaseci.utils.utils import logger

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
    try:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=2048)]
        )
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
module = hub.load(
    "https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3"
)


@jaseci_action(act_group=["use"], aliases=["enc_question"], allow_remote=True)
def question_encode(question: Union[str, list]):
    logger.info("Received request : use_qa.question_encode")
    if isinstance(question, list):
        embed = (
            module.signatures["question_encoder"](tf.constant(question))["outputs"]
            .numpy()
            .tolist()
        )
        logger.info("Returning response : use_qa.question_encode")
        return embed
    elif isinstance(question, str):
        embed = (
            module.signatures["question_encoder"](tf.constant([question]))["outputs"]
            .numpy()
            .tolist()
        )
        logger.info("Returning response : use_qa.question_encode")
        return embed


@jaseci_action(act_group=["use"], aliases=["enc_answer"], allow_remote=True)
def answer_encode(answer: Union[str, list], context: Union[str, list] = None):
    logger.info("Received request : use_qa.answer_encode")
    if context is None:
        context = answer
    if isinstance(answer, list):
        embed = (
            module.signatures["response_encoder"](
                input=tf.constant(answer), context=tf.constant(context)
            )["outputs"]
            .numpy()
            .tolist()
        )
        logger.info("Returning response : use_qa.answer_encode")
        return embed
    elif isinstance(answer, str):
        embed = (
            module.signatures["response_encoder"](
                input=tf.constant([answer]), context=tf.constant([context])
            )["outputs"]
            .numpy()
            .tolist()
        )
        logger.info("Returning response : use_qa.answer_encode")
        return embed


@jaseci_action(act_group=["use"], allow_remote=True)
def cos_sim_score(q_emb: list, a_emb: list):
    logger.info("Received request : use_qa.cos_sim_score")
    norm = np.linalg.norm
    score = np.dot(q_emb, a_emb) / (norm(q_emb) * norm(a_emb))
    logger.info("Returning response : use_qa.cos_sim_score")
    return score


@jaseci_action(act_group=["use"], aliases=["qa_score"], allow_remote=True)
def dist_score(q_emb: list, a_emb: list):
    logger.info("Received request : use_qa.dist_score")
    score = np.inner(q_emb, a_emb).tolist()
    logger.info("Returning response : use_qa.dist_score")
    return score


@jaseci_action(act_group=["use"], allow_remote=True)
def question_similarity(text1: str, text2: str):
    logger.info("Received request : use_qa.question_similarity")
    enc_a = np.squeeze(np.asarray(question_encode(text1)))
    enc_b = np.squeeze(np.asarray(question_encode(text2)))
    score = cos_sim_score(list(enc_a), list(enc_b))
    logger.info("Returning response : use_qa.question_similarity")
    return score


@jaseci_action(act_group=["use"], allow_remote=True)
def question_classify(text: str, classes: list):
    logger.info("Received request : use_qa.question_classify")
    text_emb = np.squeeze(np.asarray(question_encode(text)))
    ret = {"match": "", "match_idx": -1, "scores": []}
    for i in classes:
        i_emb = np.squeeze(np.asarray(question_encode(i))) if isinstance(i, str) else i
        ret["scores"].append(cos_sim_score(text_emb, i_emb))
    top_hit = ret["scores"].index(max(ret["scores"]))
    ret["match_idx"] = top_hit
    ret["match"] = (
        classes[top_hit] if isinstance(classes[top_hit], str) else "[embedded value]"
    )
    logger.info("Returning response : use_qa.question_classify")
    return ret


@jaseci_action(act_group=["use"], allow_remote=True)
def answer_similarity(text1: str, text2: str):
    logger.info("Received request : use_qa.answer_similarity")
    enc_a = np.squeeze(np.asarray(answer_encode(text1)))
    enc_b = np.squeeze(np.asarray(answer_encode(text2)))
    score = cos_sim_score(list(enc_a), list(enc_b))
    logger.info("Returning response : use_qa.answer_similarity")
    return score


@jaseci_action(act_group=["use"], allow_remote=True)
def answer_classify(text: str, classes: list):
    logger.info("Received request : use_qa.answer_classify")
    text_emb = np.squeeze(np.asarray(answer_encode(text)))
    ret = {"match": "", "match_idx": -1, "scores": []}
    for i in classes:
        i_emb = np.squeeze(np.asarray(answer_encode(i))) if isinstance(i, str) else i
        ret["scores"].append(cos_sim_score(text_emb, i_emb))
    top_hit = ret["scores"].index(max(ret["scores"]))
    ret["match_idx"] = top_hit
    ret["match"] = (
        classes[top_hit] if isinstance(classes[top_hit], str) else "[embedded value]"
    )
    logger.info("Returning response : use_qa.answer_classify")
    return ret


@jaseci_action(act_group=["use"], allow_remote=True)
def qa_similarity(text1: str, text2: str):
    logger.info("Received request : use_qa.qa_similarity")
    enc_a = np.squeeze(np.asarray(question_encode(text1)))
    enc_b = np.squeeze(np.asarray(answer_encode(text2)))
    score = cos_sim_score(list(enc_a), list(enc_b))
    logger.info("Returning response : use_qa.qa_similarity")
    return score


@jaseci_action(act_group=["use"], allow_remote=True)
def qa_classify(text: str, classes: list):
    logger.info("Received request : use_qa.qa_classify")
    text_emb = np.squeeze(np.asarray(question_encode(text)))
    ret = {"match": "", "match_idx": -1, "scores": []}
    for i in classes:
        i_emb = np.squeeze(np.asarray(answer_encode(i))) if isinstance(i, str) else i
        ret["scores"].append(cos_sim_score(text_emb, i_emb))
    top_hit = ret["scores"].index(max(ret["scores"]))
    ret["match_idx"] = top_hit
    ret["match"] = (
        classes[top_hit] if isinstance(classes[top_hit], str) else "[embedded value]"
    )
    logger.info("Returning response : use_qa.qa_classify")
    return ret


if __name__ == "__main__":
    from jaseci.actions.remote_actions import launch_server

    launch_server(port=8000)
