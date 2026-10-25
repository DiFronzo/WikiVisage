import importlib

import config
import face_client
import liftwing_client


def test_every_outbound_client_sends_the_same_user_agent(flask_app):
    app_module = importlib.import_module("app")
    worker_module = importlib.import_module("worker")

    assert config.USER_AGENT.startswith(f"WikiVisage/{config.PROJECT_VERSION} (")
    assert app_module.USER_AGENT == config.USER_AGENT
    assert worker_module.USER_AGENT == config.USER_AGENT
    assert face_client.USER_AGENT == config.USER_AGENT
    assert liftwing_client.USER_AGENT == config.USER_AGENT
