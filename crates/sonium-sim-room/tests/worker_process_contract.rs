use sonium_sim_room::fem_runner::{
    FemRunCheckpoint, FemRunStatus, FemWorkerProcess, FemWorkerProcessGroup, FemWorkerProcessSpec,
    FemWorkerTaskCoordinator, ResourceBudget, WorkloadShape, plan_solve_schedule,
};

fn schedule() -> sonium_sim_room::fem_runner::SolveSchedule {
    plan_solve_schedule(
        WorkloadShape {
            candidate_count: 1,
            frequency_count: 2,
            source_count: 1,
            rhs_block_size: 1,
        },
        ResourceBudget {
            lane_limit: 1,
            memory_budget_bytes: 1024,
            shared_bytes: 1,
            per_lane_bytes: 1,
            per_frequency_bytes: 1,
            per_rhs_bytes: 1,
            per_output_bytes: 1,
        },
    )
    .unwrap()
}

fn fixture_spec(mode: &str) -> FemWorkerProcessSpec {
    fixture_spec_with_count(mode, 2)
}

fn fixture_spec_with_count(mode: &str, task_count: usize) -> FemWorkerProcessSpec {
    FemWorkerProcessSpec::new(env!("CARGO_BIN_EXE_roomsim_worker_protocol_fixture"))
        .env("ROOMSIM_WORKER_TASK_COUNT", task_count.to_string())
        .env("ROOMSIM_WORKER_FIXTURE_MODE", mode)
}

fn group_schedule() -> sonium_sim_room::fem_runner::SolveSchedule {
    plan_solve_schedule(
        WorkloadShape {
            candidate_count: 2,
            frequency_count: 1,
            source_count: 1,
            rhs_block_size: 1,
        },
        ResourceBudget {
            lane_limit: 2,
            memory_budget_bytes: 1024,
            shared_bytes: 1,
            per_lane_bytes: 1,
            per_frequency_bytes: 1,
            per_rhs_bytes: 1,
            per_output_bytes: 1,
        },
    )
    .unwrap()
}

#[test]
fn process_worker_round_trip_releases_checked_responses_to_coordinator() {
    let schedule = schedule();
    let mut coordinator = FemWorkerTaskCoordinator::new(&schedule, 1).unwrap();
    let mut worker = FemWorkerProcess::spawn(&fixture_spec("reverse"), 0, 1).unwrap();
    let responses = worker.exchange(&coordinator).unwrap();
    assert_eq!(responses.len(), 2);
    let mut ready_counts = Vec::new();
    for response in responses {
        let ready = coordinator.accept_response(response).unwrap();
        ready_counts.push(ready.len());
    }
    assert_eq!(ready_counts, [0, 2]);
    assert!(worker.finish().unwrap().success());
}

#[test]
fn process_worker_failure_response_reaches_coordinator() {
    let schedule = schedule();
    let mut coordinator = FemWorkerTaskCoordinator::new(&schedule, 1).unwrap();
    let mut worker = FemWorkerProcess::spawn(&fixture_spec("failed"), 0, 1).unwrap();
    let responses = worker.exchange(&coordinator).unwrap();
    assert_eq!(responses.len(), 2);
    let error = coordinator
        .accept_response(responses[0].clone())
        .unwrap_err();
    assert!(error.to_string().contains("fixture worker failure"));
    assert!(worker.finish().unwrap().success());
}

#[test]
fn process_worker_truncated_response_fails_and_process_is_reaped() {
    let schedule = schedule();
    let coordinator = FemWorkerTaskCoordinator::new(&schedule, 1).unwrap();
    let mut worker = FemWorkerProcess::spawn(&fixture_spec("truncate"), 0, 1).unwrap();
    let error = worker.exchange(&coordinator).unwrap_err();
    assert!(error.to_string().contains("worker frame") || error.to_string().contains("I/O"));
    let _ = worker.terminate().unwrap();
}

#[test]
fn process_worker_nonzero_exit_is_reported() {
    let worker = FemWorkerProcess::spawn(&fixture_spec("exit_nonzero"), 0, 1).unwrap();
    let error = worker.finish().unwrap_err();
    assert!(error.to_string().contains("exited unsuccessfully"));
}

#[test]
fn process_worker_rejects_misrouted_response_before_commit() {
    let schedule = schedule();
    let coordinator = FemWorkerTaskCoordinator::new(&schedule, 1).unwrap();
    let mut worker = FemWorkerProcess::spawn(&fixture_spec("wrong_worker"), 0, 1).unwrap();
    let error = worker.exchange(&coordinator).unwrap_err();
    assert!(error.to_string().contains("worker"));
}

#[test]
fn process_worker_rejects_invalid_spec_and_worker_index() {
    assert!(FemWorkerProcess::spawn(&FemWorkerProcessSpec::new(""), 0, 1).is_err());
    assert!(FemWorkerProcess::spawn(&fixture_spec("ok"), 1, 1).is_err());
    assert!(FemWorkerProcess::spawn(&fixture_spec("ok"), 0, 0).is_err());
}

#[test]
fn process_group_exchanges_two_rank_work_and_finishes_all_ranks() {
    let schedule = group_schedule();
    let mut coordinator = FemWorkerTaskCoordinator::new(&schedule, 2).unwrap();
    let mut group = FemWorkerProcessGroup::spawn(vec![
        fixture_spec_with_count("ok", 1),
        fixture_spec_with_count("ok", 1),
    ])
    .unwrap();
    assert_eq!(group.worker_count(), 2);
    let responses = group.exchange_all(&coordinator).unwrap();
    assert_eq!(responses.len(), 2);
    for response in responses {
        assert_eq!(coordinator.accept_response(response).unwrap().len(), 1);
    }
    group.finish_all().unwrap();
}

#[test]
fn process_group_restarts_one_rank_before_dispatch() {
    let schedule = group_schedule();
    let mut coordinator = FemWorkerTaskCoordinator::new(&schedule, 2).unwrap();
    let mut group = FemWorkerProcessGroup::spawn(vec![
        fixture_spec_with_count("exit_nonzero", 1),
        fixture_spec_with_count("ok", 1),
    ])
    .unwrap();
    group
        .restart_worker(0, &fixture_spec_with_count("ok", 1))
        .unwrap();
    let responses = group.exchange_all(&coordinator).unwrap();
    assert_eq!(responses.len(), 2);
    for response in responses {
        coordinator.accept_response(response).unwrap();
    }
    group.finish_all().unwrap();
}

#[test]
fn process_group_terminates_sibling_ranks_after_transport_failure() {
    let schedule = group_schedule();
    let coordinator = FemWorkerTaskCoordinator::new(&schedule, 2).unwrap();
    let mut group = FemWorkerProcessGroup::spawn(vec![
        fixture_spec_with_count("ok", 1),
        fixture_spec_with_count("truncate", 1),
    ])
    .unwrap();
    let error = group.exchange_all(&coordinator).unwrap_err();
    assert!(error.to_string().contains("worker frame") || error.to_string().contains("I/O"));
}

#[test]
fn process_group_commits_checkpoint_and_completes_only_after_all_rank_responses() {
    let schedule = group_schedule();
    let mut checkpoint = FemRunCheckpoint::new(&schedule, 1024).unwrap();
    let mut group = FemWorkerProcessGroup::spawn(vec![
        fixture_spec_with_count("ok", 1),
        fixture_spec_with_count("ok", 1),
    ])
    .unwrap();
    let report = group
        .execute_checkpointed(&schedule, &mut checkpoint)
        .unwrap();
    assert_eq!(report.status, FemRunStatus::Completed);
    assert_eq!(report.committed_task_count, 2);
    assert_eq!(checkpoint.status, FemRunStatus::Completed);
    assert_eq!(checkpoint.next_task_index, 2);
    group.finish_all().unwrap();
}

#[test]
fn process_group_durably_cancels_checkpoint_on_rank_transport_failure() {
    let schedule = group_schedule();
    let mut checkpoint = FemRunCheckpoint::new(&schedule, 1024).unwrap();
    let mut group = FemWorkerProcessGroup::spawn(vec![
        fixture_spec_with_count("ok", 1),
        fixture_spec_with_count("truncate", 1),
    ])
    .unwrap();
    let error = group
        .execute_checkpointed(&schedule, &mut checkpoint)
        .unwrap_err();
    assert!(error.to_string().contains("worker frame") || error.to_string().contains("I/O"));
    assert_eq!(checkpoint.status, FemRunStatus::Cancelled);
    assert!(checkpoint.cancellation_reason.is_some());
}

#[test]
fn process_group_replaces_failed_ranks_and_resumes_cancelled_checkpoint() {
    let schedule = group_schedule();
    let mut checkpoint = FemRunCheckpoint::new(&schedule, 1024).unwrap();
    let mut group = FemWorkerProcessGroup::spawn(vec![
        fixture_spec_with_count("ok", 1),
        fixture_spec_with_count("truncate", 1),
    ])
    .unwrap();
    group
        .execute_checkpointed(&schedule, &mut checkpoint)
        .unwrap_err();
    assert_eq!(checkpoint.status, FemRunStatus::Cancelled);

    let cursor = checkpoint.next_task_index;
    checkpoint.resume(&schedule, cursor).unwrap();
    group
        .restart_worker(0, &fixture_spec_with_count("ok", 1))
        .unwrap();
    group
        .restart_worker(1, &fixture_spec_with_count("ok", 1))
        .unwrap();
    let report = group
        .execute_checkpointed(&schedule, &mut checkpoint)
        .unwrap();
    assert_eq!(report.status, FemRunStatus::Completed);
    assert_eq!(report.committed_task_count, 2);
    group.finish_all().unwrap();
}
