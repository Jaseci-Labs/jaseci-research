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
module = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


@jaseci_action(act_group=["use"], aliases=["get_embedding"], allow_remote=True)
def encode(text: Union[str, list]):
    logger.info("Received request : use_enc.encode")
    if isinstance(text, str):
        text = [text]
    score = module(text).numpy().tolist()
    logger.info("Returning response : use_enc.encode")
    return score


@jaseci_action(act_group=["use"], allow_remote=True)
def cos_sim_score(q_emb: list, a_emb: list):
    logger.info("Received request : use_enc.cos_sim_score")
    norm = np.linalg.norm
    result = np.dot(q_emb, a_emb) / (norm(q_emb) * norm(a_emb))
    logger.info("Returning response : use_enc.cos_sim_score")
    return float(result.astype(float))


@jaseci_action(act_group=["use"], allow_remote=True)
def text_similarity(text1: str, text2: str):
    logger.info("Received request : use_enc.text_similarity")
    enc_a = np.squeeze(np.asarray(encode(text1)))
    enc_b = np.squeeze(np.asarray(encode(text2)))
    score = cos_sim_score(list(enc_a), list(enc_b))
    logger.info("Returning response : use_enc.text_similarity")
    return score


@jaseci_action(act_group=["use"], allow_remote=True)
def text_classify(text: str, classes: list):
    logger.info("Received request : use_enc.text_classify")
    text_emb = np.squeeze(np.asarray(encode(text)))
    ret = {"match": "", "match_idx": -1, "scores": []}
    for i in classes:
        i_emb = np.squeeze(np.asarray(encode(i))) if isinstance(i, str) else i
        ret["scores"].append(cos_sim_score(text_emb, i_emb))
    top_hit = ret["scores"].index(max(ret["scores"]))
    ret["match_idx"] = top_hit
    ret["match"] = (
        classes[top_hit] if isinstance(classes[top_hit], str) else "[embedded value]"
    )
    logger.info("Returning response : use_enc.text_classify")
    return ret


if __name__ == "__main__":
    from jaseci.actions.remote_actions import launch_server

    launch_server(port=8000)
