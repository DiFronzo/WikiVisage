"""Tests for the web app: membership."""

from tests._app_preamble import (
    app_module,
)
from tests._app_support import (
    _flashes_chunk6,
    _make_authed_client,
    _make_invite_code_collision_error,
    _project_settings_base_row,
    _set_csrf_chunk6,
)


def test_remove_member_success(monkeypatch, fake_user):
    project = _project_settings_base_row()
    banned = {"called": False}

    def route_execute(sql, params, fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "UPDATE project_members SET status = 'banned'" in sql:
            banned["called"] = True
            assert params == (1, 5)
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings/remove-member",
        data={"csrf_token": "testtoken", "member_user_id": "5"},
    )

    assert response.status_code == 302
    assert "/project/1/settings" in response.headers["Location"]
    assert banned["called"] is True
    flashes = _flashes_chunk6(client)
    assert any("banned" in msg.lower() for _cat, msg in flashes)


def test_remove_member_csrf_fail(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)
    _set_csrf_chunk6(client, "expected")

    response = client.post(
        "/project/1/settings/remove-member",
        data={"csrf_token": "wrong", "member_user_id": "5"},
    )

    assert response.status_code == 400
    assert b"Invalid CSRF token" in response.data


def test_remove_member_project_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings/remove-member",
        data={"csrf_token": "testtoken", "member_user_id": "5"},
    )

    assert response.status_code == 404


def test_remove_member_self_removal_blocked(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings/remove-member",
        data={"csrf_token": "testtoken", "member_user_id": "1"},
    )

    assert response.status_code == 302
    flashes = _flashes_chunk6(client)
    assert any("cannot remove yourself" in msg.lower() for _cat, msg in flashes)


def test_remove_member_no_member_specified(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings/remove-member",
        data={"csrf_token": "testtoken"},
    )

    assert response.status_code == 302
    flashes = _flashes_chunk6(client)
    assert any("no member" in msg.lower() for _cat, msg in flashes)


def test_remove_member_invalid_member_id(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings/remove-member",
        data={"csrf_token": "testtoken", "member_user_id": "abc"},
    )

    assert response.status_code == 302
    flashes = _flashes_chunk6(client)
    assert any("invalid" in msg.lower() for _cat, msg in flashes)


def test_remove_member_not_found_in_db(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "UPDATE project_members SET status = 'banned'" in sql:
            return 0
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings/remove-member",
        data={"csrf_token": "testtoken", "member_user_id": "99"},
    )

    assert response.status_code == 302
    flashes = _flashes_chunk6(client)
    assert any("not found" in msg.lower() for _cat, msg in flashes)


def test_remove_member_db_error(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "UPDATE project_members SET status = 'banned'" in sql:
            raise app_module.DatabaseError("ban failed")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings/remove-member",
        data={"csrf_token": "testtoken", "member_user_id": "5"},
    )

    assert response.status_code == 302
    flashes = _flashes_chunk6(client)
    assert any("failed" in msg.lower() for _cat, msg in flashes)


def test_unban_member_success(monkeypatch, fake_user):
    project = _project_settings_base_row()
    unbanned = {"called": False}

    def route_execute(sql, params, fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "DELETE FROM project_members" in sql and "status = 'banned'" in sql:
            unbanned["called"] = True
            assert params == (1, 5)
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings/unban-member",
        data={"csrf_token": "testtoken", "member_user_id": "5"},
    )

    assert response.status_code == 302
    assert "/project/1/settings" in response.headers["Location"]
    assert unbanned["called"] is True
    flashes = _flashes_chunk6(client)
    assert any("unbanned" in msg.lower() for _cat, msg in flashes)


def test_unban_member_not_found(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "DELETE FROM project_members" in sql and "status = 'banned'" in sql:
            return 0
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings/unban-member",
        data={"csrf_token": "testtoken", "member_user_id": "99"},
    )

    assert response.status_code == 302
    flashes = _flashes_chunk6(client)
    assert any("not found" in msg.lower() for _cat, msg in flashes)


def test_unban_member_csrf_fail(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)
    _set_csrf_chunk6(client, "expected")

    response = client.post(
        "/project/1/settings/unban-member",
        data={"csrf_token": "wrong", "member_user_id": "5"},
    )

    assert response.status_code == 400
    assert b"Invalid CSRF token" in response.data


def test_unban_member_project_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings/unban-member",
        data={"csrf_token": "testtoken", "member_user_id": "5"},
    )

    assert response.status_code == 404


def test_unban_member_db_error(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "DELETE FROM project_members" in sql and "status = 'banned'" in sql:
            raise app_module.DatabaseError("unban failed")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings/unban-member",
        data={"csrf_token": "testtoken", "member_user_id": "5"},
    )

    assert response.status_code == 302
    flashes = _flashes_chunk6(client)
    assert any("failed" in msg.lower() for _cat, msg in flashes)


def test_invite_code_generate(monkeypatch, fake_user):
    project = _project_settings_base_row()
    update_calls = []

    def route_execute(sql, params, fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "UPDATE projects SET invite_code" in sql:
            update_calls.append(params)
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/invite-code",
        data={"csrf_token": "testtoken", "action": "generate"},
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/project/1/settings")
    assert len(update_calls) == 1
    assert update_calls[0][0] is not None
    assert len(update_calls[0][0]) == 8
    flashes = _flashes_chunk6(client)
    assert any("generated" in msg.lower() for _cat, msg in flashes)


def test_invite_code_revoke(monkeypatch, fake_user):
    project = _project_settings_base_row()
    update_calls = []

    def route_execute(sql, params, fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "UPDATE projects SET invite_code = NULL" in sql:
            update_calls.append(params)
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/invite-code",
        data={"csrf_token": "testtoken", "action": "revoke"},
    )

    assert response.status_code == 302
    assert len(update_calls) == 1
    flashes = _flashes_chunk6(client)
    assert any("revoked" in msg.lower() for _cat, msg in flashes)


def test_invite_code_invalid_action(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/invite-code",
        data={"csrf_token": "testtoken", "action": "bogus"},
    )

    assert response.status_code == 302
    flashes = _flashes_chunk6(client)
    assert any("invalid" in msg.lower() for _cat, msg in flashes)


def test_invite_code_project_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/invite-code",
        data={"csrf_token": "testtoken", "action": "generate"},
    )

    assert response.status_code == 404


def test_invite_code_csrf_fail(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)

    response = client.post("/project/1/invite-code", data={"csrf_token": "wrong"})

    assert response.status_code == 400


def test_invite_code_generate_db_error(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "UPDATE projects SET invite_code" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/invite-code",
        data={"csrf_token": "testtoken", "action": "generate"},
    )

    assert response.status_code == 302
    flashes = _flashes_chunk6(client)
    assert any("failed" in msg.lower() for _cat, msg in flashes)


def test_invite_code_generate_retries_on_collision(monkeypatch, fake_user):
    """First UPDATE raises an invite_code collision; second attempt succeeds."""
    project = _project_settings_base_row()
    call_count = {"n": 0}

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "UPDATE projects SET invite_code" in sql:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise _make_invite_code_collision_error()
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/invite-code",
        data={"csrf_token": "testtoken", "action": "generate"},
    )

    assert response.status_code == 302
    assert call_count["n"] == 2
    flashes = _flashes_chunk6(client)
    assert any("generated" in msg.lower() for _cat, msg in flashes)


def test_invite_code_generate_fails_after_max_retries(monkeypatch, fake_user):
    """All retry attempts raise invite_code collisions; route shows failure flash."""
    project = _project_settings_base_row()

    def route_execute_fail(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "UPDATE projects SET invite_code" in sql:
            raise _make_invite_code_collision_error()
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute_fail)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/invite-code",
        data={"csrf_token": "testtoken", "action": "generate"},
    )

    assert response.status_code == 302
    flashes = _flashes_chunk6(client)
    assert any("failed" in msg.lower() for _cat, msg in flashes)

    insert_calls = []

    def route_execute(sql, params, fetch):
        if "WHERE p.invite_code = %s" in sql:
            return [{"id": 42, "label": "Cool Project", "user_id": 99, "wiki_username": "Alice"}]
        if "SELECT status FROM project_members" in sql:
            return ()
        if "INSERT INTO project_members" in sql:
            insert_calls.append(params)
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/join",
        data={"csrf_token": "testtoken", "invite_code": "ABC12345"},
    )

    assert response.status_code == 302
    assert "/project/42" in response.headers["Location"]
    assert len(insert_calls) == 1
    flashes = _flashes_chunk6(client)
    assert any("joined" in msg.lower() for _cat, msg in flashes)


def test_join_by_code_invalid_code(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "WHERE p.invite_code = %s" in sql:
            return ()
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/join",
        data={"csrf_token": "testtoken", "invite_code": "BADCODE1"},
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")
    flashes = _flashes_chunk6(client)
    assert any("invalid" in msg.lower() for _cat, msg in flashes)


def test_join_by_code_empty_code(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)
    _set_csrf_chunk6(client)

    response = client.post(
        "/join",
        data={"csrf_token": "testtoken", "invite_code": ""},
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")
    flashes = _flashes_chunk6(client)
    assert any("enter" in msg.lower() for _cat, msg in flashes)


def test_join_by_code_banned_user(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "WHERE p.invite_code = %s" in sql:
            return [{"id": 42, "label": "Cool Project", "user_id": 99, "wiki_username": "Alice"}]
        if "SELECT status FROM project_members" in sql:
            return [{"status": "banned"}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/join",
        data={"csrf_token": "testtoken", "invite_code": "ABC12345"},
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")
    flashes = _flashes_chunk6(client)
    assert any("banned" in msg.lower() for _cat, msg in flashes)


def test_join_by_code_already_member(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "WHERE p.invite_code = %s" in sql:
            return [{"id": 42, "label": "Cool Project", "user_id": 99, "wiki_username": "Alice"}]
        if "SELECT status FROM project_members" in sql:
            return [{"status": "active"}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/join",
        data={"csrf_token": "testtoken", "invite_code": "ABC12345"},
    )

    assert response.status_code == 302
    assert "/project/42" in response.headers["Location"]
    flashes = _flashes_chunk6(client)
    assert any("already" in msg.lower() for _cat, msg in flashes)


def test_join_by_code_owner(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "WHERE p.invite_code = %s" in sql:
            return [{"id": 42, "label": "My Project", "user_id": 1, "wiki_username": "tester"}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/join",
        data={"csrf_token": "testtoken", "invite_code": "ABC12345"},
    )

    assert response.status_code == 302
    assert "/project/42" in response.headers["Location"]
    flashes = _flashes_chunk6(client)
    assert any("owner" in msg.lower() for _cat, msg in flashes)


def test_join_by_code_csrf_fail(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)

    response = client.post("/join", data={"csrf_token": "wrong"})

    assert response.status_code == 400


def test_join_by_code_db_error_on_lookup(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "WHERE p.invite_code = %s" in sql:
            raise app_module.DatabaseError("lookup failed")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/join",
        data={"csrf_token": "testtoken", "invite_code": "ABC12345"},
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")
    flashes = _flashes_chunk6(client)
    assert any("wrong" in msg.lower() for _cat, msg in flashes)


def test_leave_project_success(monkeypatch, fake_user):
    project = _project_settings_base_row()
    project["user_id"] = 99
    deleted = {"called": False}

    def route_execute(sql, params, fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [project.copy()]
        if "DELETE FROM project_members" in sql:
            deleted["called"] = True
            assert params == (1, 1)
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/project/1/leave", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert "/dashboard" in response.headers["Location"]
    assert deleted["called"] is True
    flashes = _flashes_chunk6(client)
    assert any("left the project" in msg.lower() for _cat, msg in flashes)


def test_leave_project_owner_blocked(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [project.copy()]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/project/1/leave", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert "/project/1" in response.headers["Location"]
    flashes = _flashes_chunk6(client)
    assert any("cannot leave" in msg.lower() for _cat, msg in flashes)


def test_leave_project_csrf_fail(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)
    _set_csrf_chunk6(client, "expected")

    response = client.post("/project/1/leave", data={"csrf_token": "wrong"})

    assert response.status_code == 400
    assert b"Invalid CSRF token" in response.data


def test_leave_project_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/project/1/leave", data={"csrf_token": "testtoken"})

    assert response.status_code == 404


def test_leave_project_not_a_member(monkeypatch, fake_user):
    project = _project_settings_base_row()
    project["user_id"] = 99

    def route_execute(sql, _params, fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [project.copy()]
        if "DELETE FROM project_members" in sql:
            return 0
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/project/1/leave", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert "/dashboard" in response.headers["Location"]
    flashes = _flashes_chunk6(client)
    assert any("not a member" in msg.lower() for _cat, msg in flashes)


def test_leave_project_db_error(monkeypatch, fake_user):
    project = _project_settings_base_row()
    project["user_id"] = 99

    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [project.copy()]
        if "DELETE FROM project_members" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/project/1/leave", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert "/project/1" in response.headers["Location"]
    flashes = _flashes_chunk6(client)
    assert any("failed to leave" in msg.lower() for _cat, msg in flashes)


def test_invite_code_generate_sql_includes_created_at(monkeypatch, fake_user):
    project = _project_settings_base_row()
    captured_sql = []

    def route_execute(sql, params, fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "UPDATE projects SET invite_code" in sql:
            captured_sql.append(sql)
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    client.post(
        "/project/1/invite-code",
        data={"csrf_token": "testtoken", "action": "generate"},
    )

    assert len(captured_sql) == 1
    assert "invite_code_created_at = NOW()" in captured_sql[0]


def test_invite_code_revoke_sql_clears_created_at(monkeypatch, fake_user):
    project = _project_settings_base_row()
    captured_sql = []

    def route_execute(sql, params, fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "UPDATE projects SET invite_code = NULL" in sql:
            captured_sql.append(sql)
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    client.post(
        "/project/1/invite-code",
        data={"csrf_token": "testtoken", "action": "revoke"},
    )

    assert len(captured_sql) == 1
    assert "invite_code_created_at = NULL" in captured_sql[0]


def test_join_by_code_expired_shows_invalid(monkeypatch, fake_user):
    def route_execute(sql, params, _fetch):
        if "WHERE p.invite_code = %s" in sql:
            assert "invite_code_created_at" in sql
            assert "INTERVAL %s DAY" in sql
            assert params == ("ABC12345", app_module._INVITE_CODE_TTL_DAYS)
            return ()
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/join",
        data={"csrf_token": "testtoken", "invite_code": "ABC12345"},
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")
    flashes = _flashes_chunk6(client)
    assert any("invalid" in msg.lower() for _cat, msg in flashes)


def test_invite_code_ttl_days_is_seven():
    assert app_module._INVITE_CODE_TTL_DAYS == 7
