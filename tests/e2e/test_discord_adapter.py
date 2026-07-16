"""Minimal e2e tests for Discord mention stripping + /command detection.

Covers the fix for slash commands not being recognized when sent via
@mention in a channel, especially after auto-threading.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.e2e.conftest import (
    BOT_USER_ID,
    E2E_MESSAGE_SETTLE_DELAY,
    get_response_text,
    make_discord_message,
    make_fake_dm_channel,
    make_fake_thread,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _hermetic_discord_env(monkeypatch):
    """Keep live profile channel policy from contaminating adapter unit tests."""
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "*")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "*")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    monkeypatch.setenv("DISCORD_HISTORY_BACKFILL_LIMIT", "0")
    monkeypatch.delenv("DISCORD_IGNORED_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)


async def dispatch(adapter, msg):
    await adapter._handle_message(msg)
    await asyncio.sleep(E2E_MESSAGE_SETTLE_DELAY)


class TestPlatformVerificationContext:
    async def test_guild_message_attaches_verified_role_metadata(
        self, discord_adapter, bot_user, monkeypatch
    ):
        monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
        monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "*")
        monkeypatch.setenv("DISCORD_FREE_RESPONSE_CHANNELS", "*")
        monkeypatch.delenv("DISCORD_IGNORED_CHANNELS", raising=False)
        discord_adapter.handle_message = AsyncMock()
        discord_adapter._allowed_user_ids = {"11111"}
        discord_adapter._text_batch_delay_seconds = 0
        author = SimpleNamespace(
            id=11111,
            name="testuser",
            display_name="Display Name",
            bot=False,
            roles=[
                SimpleNamespace(id=700, name="Developer"),
                SimpleNamespace(id=701, name="QA"),
            ],
        )
        msg = make_discord_message(
            content=f"<@{BOT_USER_ID}> approve the Planned transition for IOIA-5000",
            author=author,
            mentions=[bot_user],
            message_id=88001,
        )

        await discord_adapter._handle_message(msg)

        discord_adapter.handle_message.assert_awaited_once()
        event = discord_adapter.handle_message.await_args.args[0]
        assert event.metadata["platform_verification_context"] == {
            "schema_version": 1,
            "platform": "discord",
            "verification_source": "platform_adapter",
            "scope_id": str(msg.guild.id),
            "channel_id": str(msg.channel.id),
            "thread_id": None,
            "message_id": "88001",
            "user_id": "11111",
            "user_name": "Display Name",
            "roles": [
                {"id": "700", "name": "Developer"},
                {"id": "701", "name": "QA"},
            ],
            "message_text": "approve the Planned transition for IOIA-5000",
        }

    async def test_dm_message_has_no_scope_or_roles(self, discord_adapter):
        discord_adapter.handle_message = AsyncMock()
        discord_adapter._allowed_user_ids = {"11111"}
        discord_adapter._text_batch_delay_seconds = 0
        dm = make_fake_dm_channel()
        author = SimpleNamespace(
            id=11111,
            name="testuser",
            display_name="DM User",
            bot=False,
            roles=[SimpleNamespace(id=700, name="Developer")],
        )
        msg = make_discord_message(
            content="approve the draft creation",
            author=author,
            channel=dm,
            mentions=[],
            message_id=88002,
        )

        await discord_adapter._handle_message(msg)

        event = discord_adapter.handle_message.await_args.args[0]
        context = event.metadata["platform_verification_context"]
        assert context["scope_id"] is None
        assert context["roles"] == []
        assert context["message_text"] == "approve the draft creation"


class TestMentionStrippedCommandDispatch:
    async def test_mention_then_command(self, discord_adapter, bot_user):
        """<@BOT> /help → mention stripped, /help dispatched."""
        msg = make_discord_message(
            content=f"<@{BOT_USER_ID}> /help",
            mentions=[bot_user],
        )
        await dispatch(discord_adapter, msg)
        response = get_response_text(discord_adapter)
        assert response is not None
        assert "/new" in response

    async def test_nickname_mention_then_command(self, discord_adapter, bot_user):
        """<@!BOT> /help → nickname mention also stripped, /help works."""
        msg = make_discord_message(
            content=f"<@!{BOT_USER_ID}> /help",
            mentions=[bot_user],
        )
        await dispatch(discord_adapter, msg)
        response = get_response_text(discord_adapter)
        assert response is not None
        assert "/new" in response

    async def test_text_before_command_not_detected(self, discord_adapter, bot_user):
        """'<@BOT> something else /help' → mention stripped, but 'something else /help'
        doesn't start with / so it's treated as text, not a command."""
        msg = make_discord_message(
            content=f"<@{BOT_USER_ID}> something else /help",
            mentions=[bot_user],
        )
        await dispatch(discord_adapter, msg)
        # Message is accepted (not dropped by mention gate), but since it doesn't
        # start with / it's routed as text — no command output, and no agent in this
        # mock setup means no send call either.
        response = get_response_text(discord_adapter)
        assert response is None or "/new" not in response

    async def test_no_mention_in_channel_dropped(self, discord_adapter):
        """Message without @mention in server channel → silently dropped."""
        msg = make_discord_message(content="/help", mentions=[])
        await dispatch(discord_adapter, msg)
        assert get_response_text(discord_adapter) is None

    async def test_dm_no_mention_needed(self, discord_adapter):
        """DMs don't require @mention — /help works directly."""
        dm = make_fake_dm_channel()
        msg = make_discord_message(content="/help", channel=dm, mentions=[])
        await dispatch(discord_adapter, msg)
        response = get_response_text(discord_adapter)
        assert response is not None
        assert "/new" in response


class TestAutoThreadingPreservesCommand:
    async def test_command_detected_after_auto_thread(self, discord_adapter, bot_user, monkeypatch):
        """@mention /help in channel with auto-thread → thread created AND command dispatched."""
        monkeypatch.setenv("DISCORD_AUTO_THREAD", "true")
        fake_thread = make_fake_thread(thread_id=90001, name="help")
        msg = make_discord_message(
            content=f"<@{BOT_USER_ID}> /help",
            mentions=[bot_user],
        )

        # Simulate discord.py restoring the original raw content (with mention)
        # after create_thread(), which undoes any prior mention stripping.
        original_content = msg.content

        async def clobber_content(**kwargs):
            msg.content = original_content
            return fake_thread

        msg.create_thread = AsyncMock(side_effect=clobber_content)
        await dispatch(discord_adapter, msg)

        msg.create_thread.assert_awaited_once()
        response = get_response_text(discord_adapter)
        assert response is not None
        assert "/new" in response


class TestRepliedToMediaDispatch:
    async def test_reply_to_image_message_caches_referenced_attachment(
        self, discord_adapter, bot_user, monkeypatch
    ):
        """A text reply to an image-bearing Discord message should give the agent that image."""
        cached_path = "/tmp/replied-discord-image.png"

        async def fake_cache_image_from_url(url, *, ext=".jpg"):
            assert url == "https://cdn.discordapp.com/attachments/image.png"
            assert ext == ".png"
            return cached_path

        monkeypatch.setattr(
            "plugins.platforms.discord.adapter.cache_image_from_url",
            fake_cache_image_from_url,
        )
        discord_adapter.handle_message = AsyncMock()

        attachment = SimpleNamespace(
            content_type="image/png",
            filename="image.png",
            url="https://cdn.discordapp.com/attachments/image.png",
            size=1234,
        )
        referenced_message = SimpleNamespace(
            id=12345,
            content="",
            attachments=[attachment],
        )
        msg = make_discord_message(
            content=f"<@{BOT_USER_ID}> what's in this image?",
            mentions=[bot_user],
        )
        msg.type = 19
        msg.reference = SimpleNamespace(message_id=12345, resolved=referenced_message)

        await discord_adapter._handle_message(msg)

        discord_adapter.handle_message.assert_awaited_once()
        await_args = discord_adapter.handle_message.await_args
        assert await_args is not None
        event = await_args.args[0]
        assert event.reply_to_message_id == "12345"
        assert event.media_urls == [cached_path]
        assert event.media_types == ["image/png"]
        assert event.message_type.value == "photo"
