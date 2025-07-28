# Copyright 2024 Redpanda Data, Inc.
#
# Use of this software is governed by the Business Source License
# included in the file licenses/BSL.md
#
# As of the Change Date specified in that file, in accordance with
# the Business Source License, use of this software will be governed
# by the Apache License, Version 2.0

import time

import requests
from ducktape.utils.util import wait_until
from requests.exceptions import ConnectionError

from rptest.clients.default import DefaultClient
from rptest.clients.types import TopicSpec
from rptest.services.admin import (
    InboundDataMigration,
    InboundTopic,
    MigrationAction,
    NamespacedTopic,
    OutboundDataMigration,
)
from rptest.services.redpanda import RedpandaService
from rptest.util import bg_thread_cm

from typing import NamedTuple, List


class RpAndMigration(NamedTuple):
    redpanda: RedpandaService
    migration_id: int
    name: str


def now():
    return int(time.time() * 1000)


class DataMigrationTestMixin:
    def wait_partitions_appear(
        self, topics: list[TopicSpec], redpanda: RedpandaService | None = None
    ):
        if redpanda is None:
            redpanda = self.redpanda
        client = DefaultClient(redpanda)

        # we may be unlucky to query a slow node
        def topic_has_all_partitions(t: TopicSpec):
            part_cnt = len(client.describe_topic(t.name).partitions)
            redpanda.logger.debug(
                f"topic {t.name} has {part_cnt} partitions out of {t.partition_count} expected"
            )
            return t.partition_count == part_cnt

        def err_msg():
            msg = "Failed waiting for partitions to appear:\n"
            for t in topics:
                msg += f"   {t.name} expected {t.partition_count} partitions, "
                msg += (
                    f"got {len(client.describe_topic(t.name).partitions)} partitions\n"
                )
            return msg

        wait_until(
            lambda: all(topic_has_all_partitions(t) for t in topics),
            timeout_sec=90,
            backoff_sec=1,
            err_msg=err_msg,
        )

    def wait_partitions_disappear(
        self, topics: list[str], redpanda: RedpandaService | None = None
    ):
        if redpanda is None:
            redpanda = self.redpanda
        client = DefaultClient(redpanda)

        # we may be unlucky to query a slow node
        wait_until(
            lambda: all(client.describe_topic(t).partitions == [] for t in topics),
            timeout_sec=90,
            backoff_sec=1,
            err_msg=f"Failed waiting for partitions to disappear",
        )

    def get_migration(self, id, node=None, redpanda: RedpandaService | None = None):
        if redpanda is None:
            redpanda = self.redpanda

        try:
            return redpanda._admin.get_data_migration(id, node).json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            else:
                raise

    def get_migrations_map(self, node=None, redpanda: RedpandaService | None = None):
        if redpanda is None:
            redpanda = self.redpanda

        redpanda.logger.debug("calling self.admin.list_data_migrations")
        migrations = redpanda._admin.list_data_migrations(node).json()
        redpanda.logger.debug("received self.admin.list_data_migrations result")
        return {migration["id"]: migration for migration in migrations}

    def on_all_live_nodes(
        self, migration_id, predicate, redpanda: RedpandaService | None = None
    ):
        if redpanda is None:
            redpanda = self.redpanda

        success_cnt = 0
        exception_cnt = 0
        for n in redpanda.nodes:
            try:
                map = self.get_migrations_map(n, redpanda=redpanda)
                redpanda.logger.debug(f"migrations on node {n.name}: {map}")
                list_item = map[migration_id] if migration_id in map else None
                individual = self.get_migration(migration_id, n, redpanda=redpanda)

                if predicate(list_item) and predicate(individual):
                    success_cnt += 1
                else:
                    return False
            except ConnectionError:
                exception_cnt += 1
        return success_cnt > exception_cnt

    def validate_timing(self, time_before, happened_at):
        time_now = now()
        self.logger.debug(f"{time_before=}, {happened_at=}, {time_now=}")
        err_ms = 25  # allow for ntp error across nodes
        assert time_before - err_ms <= happened_at <= time_now + err_ms

    def wait_migration_appear(
        self,
        migration_id,
        assure_created_after,
        redpanda: RedpandaService | None = None,
    ):
        if redpanda is None:
            redpanda = self.redpanda

        def migration_present_on_node(m):
            if m is None:
                return False
            self.validate_timing(assure_created_after, m["created_timestamp"])
            return True

        def migration_is_present(id: int):
            return self.on_all_live_nodes(
                id, migration_present_on_node, redpanda=redpanda
            )

        wait_until(
            lambda: migration_is_present(migration_id),
            timeout_sec=30,
            backoff_sec=2,
            err_msg=f"Expected migration with id {migration_id} is present",
        )

    def create_and_wait(
        self,
        migration: InboundDataMigration | OutboundDataMigration,
        redpanda: RedpandaService | None = None,
    ):
        if redpanda is None:
            redpanda = self.redpanda

        def migration_id_if_exists():
            for n in redpanda.nodes:
                for m in redpanda._admin.list_data_migrations(n).json():
                    if m == migration:
                        return m[id]
            return None

        time_before_creation = now()
        try:
            reply = redpanda._admin.create_data_migration(migration).json()
            redpanda.logger.info(f"create migration reply: {reply}")
            migration_id = reply["id"]
        except requests.exceptions.HTTPError as e:
            maybe_id = migration_id_if_exists()
            if maybe_id is None:
                raise
            migration_id = maybe_id
            redpanda.logger.info(
                f"create migration failed but migration {migration_id} present: {e}"
            )

        self.wait_migration_appear(
            migration_id, time_before_creation, redpanda=redpanda
        )

        return migration_id

    def assure_not_deletable(
        self, id, node=None, redpanda: RedpandaService | None = None
    ):
        if redpanda is None:
            redpanda = self.redpanda

        try:
            redpanda._admin.delete_data_migration(id, node)
            assert False
        except requests.exceptions.HTTPError:
            pass

    def wait_for_migration_states(
        self,
        id: int,
        states: list[str],
        assure_completed_after: int = 0,
        redpanda: RedpandaService | None = None,
    ):
        if redpanda is None:
            redpanda = self.redpanda

        def migration_in_one_of_states_on_node(m):
            if m is None:
                return False
            completed_at = m.get("completed_timestamp")
            if m["state"] in ("finished", "cancelled"):
                self.validate_timing(assure_completed_after, completed_at)
            else:
                assert "completed_timestamp" not in m
            return m["state"] in states

        def migration_in_one_of_states():
            return self.on_all_live_nodes(
                id, migration_in_one_of_states_on_node, redpanda=redpanda
            )

        self.logger.info(f"waiting for {' or '.join(states)}")
        wait_until(
            migration_in_one_of_states,
            timeout_sec=90,
            backoff_sec=1,
            err_msg=f"Failed waiting for migration {id} to reach one of {states} states",
        )
        if all(state not in ("planned", "finished", "cancelled") for state in states):
            self.assure_not_deletable(id, redpanda=redpanda)

    def migrate_between_clusters(
        self,
        topics: list[NamespacedTopic],
        groups: list[str],
        source: RedpandaService,
        dest: RedpandaService,
        aliases: list[NamespacedTopic] | None = None,
        interleaved: bool = False,
    ) -> None:
        assert source != dest

        if aliases is not None:
            assert len(aliases) == len(topics)

        out_migration = OutboundDataMigration(topics=topics, consumer_groups=groups)

        out_migration_id = self.create_and_wait(out_migration, redpanda=source)
        source.logger.info(f"created outbound migration, id {out_migration_id}")

        out_migration = source._admin.get_data_migration(out_migration_id).json()
        assert len(out_migration["migration"]["topics"]) == len(topics)

        in_topics = []
        for i, out_topic_json in enumerate(out_migration["migration"]["topics"]):
            out_topic = NamespacedTopic(
                topic=out_topic_json["topic"], namespace=out_topic_json.get("ns")
            )
            in_topic = NamespacedTopic(
                topic=out_topic_json["remote_location"], namespace=out_topic.ns
            )
            alias = out_topic if aliases is None else aliases[i]
            in_topics.append(InboundTopic(source_topic_reference=in_topic, alias=alias))

            self.logger.debug(f"topic for inbound migration: {in_topics[-1].as_dict()}")

        in_migration = InboundDataMigration(
            topics=in_topics, consumer_groups=groups, await_communication=interleaved
        )
        in_migration_id = self.create_and_wait(in_migration, redpanda=dest)
        dest.logger.info(f"created inbound migration, id {in_migration_id}")

        src = RpAndMigration(
            redpanda=source, migration_id=out_migration_id, name="source"
        )
        dst = RpAndMigration(redpanda=dest, migration_id=in_migration_id, name="dest")

        def transition(rm: RpAndMigration, action: MigrationAction):
            rm.redpanda._admin.execute_data_migration_action(rm.migration_id, action)

        def wait_for_state(rm: RpAndMigration, states: List[str]):
            self.wait_for_migration_states(
                rm.migration_id, states, redpanda=rm.redpanda
            )
            self.logger.info(f"{rm.name} on one of {states}")

        @bg_thread_cm
        def communication_thread(
            src: RpAndMigration, dst: RpAndMigration, topic_name_map: dict[str, str]
        ):
            topic_name_map = {
                topics[i].topic: aliases[i].topic
                for i in range(len(topics))
                if aliases is not None
            }
            namespaced_topic_name_mapper = (
                lambda nt: {
                    "topic": topic_name_map.get(nt["topic"], nt["topic"]),
                    "ns": "kafka",
                }
                if nt["ns"] == "kafka"
                else nt
            )

            while (yield):
                try:
                    data = src.redpanda._admin.get_migrated_entities_status(
                        src.migration_id, True
                    ).json()

                    self.logger.info(
                        f"communicated migrated entities status from outbound migration {src.migration_id}: {data}"
                    )

                    data["ready_topics"] = [
                        namespaced_topic_name_mapper(t) for t in data["ready_topics"]
                    ]
                    for d in data.get("consumer_groups_data", []):
                        for topic_entry in d["topics"]:
                            topic_entry["topic"] = namespaced_topic_name_mapper(
                                topic_entry["topic"]
                            )

                    assert (
                        dst.redpanda._admin.put_migrated_entities_status(
                            dst.migration_id, data
                        ).status_code
                        == 200
                    )
                    self.logger.info(
                        f"communicated migrated entities status for inbound migration {dst.migration_id}: {data}"
                    )
                except Exception as e:
                    self.logger.info(f"error communicating between clusters")
                    self.logger.exception(e)
                time.sleep(0.1)

        if interleaved:
            transition(src, MigrationAction.prepare)
            wait_for_state(src, ["prepared"])

            transition(dst, MigrationAction.prepare)
            wait_for_state(dst, ["preparing", "prepared"])

            with communication_thread(src, dst, {}):
                transition(src, MigrationAction.execute)
                wait_for_state(src, ["executed"])
                wait_for_state(dst, ["prepared"])

            transition(src, MigrationAction.finish)
            transition(dst, MigrationAction.execute)
            wait_for_state(src, ["finished"])
            wait_for_state(dst, ["executed"])

            transition(dst, MigrationAction.finish)
            wait_for_state(dst, ["finished"])
        else:
            transition(src, MigrationAction.prepare)
            wait_for_state(src, ["prepared"])

            transition(src, MigrationAction.execute)
            wait_for_state(src, ["executed"])

            transition(src, MigrationAction.finish)
            wait_for_state(src, ["finished"])

            transition(dst, MigrationAction.prepare)
            wait_for_state(dst, ["prepared"])

            transition(dst, MigrationAction.execute)
            wait_for_state(dst, ["executed"])

            transition(dst, MigrationAction.finish)
            wait_for_state(dst, ["finished"])
