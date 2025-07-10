/*
 * Copyright 2024 Redpanda Data, Inc.
 *
 * Use of this software is governed by the Business Source License
 * included in the file licenses/BSL.md
 *
 * As of the Change Date specified in that file, in accordance with
 * the Business Source License, use of this software will be governed
 * by the Apache License, Version 2.0
 */
#pragma once

#include "cluster/data_migration_router.h"
#include "cluster/data_migration_rpc_service.h"
#include "cluster/errc.h"
#include "cluster/fwd.h"
#include "data_migration_irpc_frontend.h"

#include <system_error>

namespace cluster::data_migrations {

class service_handler : public data_migrations_service {
public:
    explicit service_handler(
      ss::scheduling_group,
      ss::smp_service_group,
      ss::sharded<frontend>&,
      ss::sharded<irpc_frontend>&,
      ss::sharded<router>&);

    ss::future<create_migration_reply>
    create_migration(create_migration_request, ::rpc::streaming_context&) final;

    ss::future<update_migration_state_reply> update_migration_state(
      update_migration_state_request, ::rpc::streaming_context&) final;

    ss::future<remove_migration_reply>
    remove_migration(remove_migration_request, ::rpc::streaming_context&) final;

    ss::future<check_ntp_states_reply>
    check_ntp_states(check_ntp_states_request, ::rpc::streaming_context&) final;

    ss::future<get_group_offsets_reply> get_group_offsets(
      get_group_offsets_request, ::rpc::streaming_context&) final;

    ss::future<set_group_offsets_reply> set_group_offsets(
      set_group_offsets_request, ::rpc::streaming_context&) final;

private:
    static cluster::errc map_error_code(std::error_code);

    ss::sharded<frontend>& _frontend;
    ss::sharded<irpc_frontend>& _irpc_frontend;
    ss::sharded<router>& _router;
};

} // namespace cluster::data_migrations
