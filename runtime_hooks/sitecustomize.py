"""Load launcher-controlled runtime hooks in every Python and Ray worker."""


def _install_runtime_hooks() -> None:
    """Stop interpreter startup if required training instrumentation cannot load."""

    try:
        from src.agents.rollout_seed import (
            install_from_environment as install_rollout_seed,
        )
        from src.agents.update_probe import install_from_environment as install_update_probe
        from src.agents.vllm_compat import install_from_environment as install_vllm_compat

        install_update_probe()
        install_vllm_compat()
        install_rollout_seed()
        # Keep the legacy startup path free of screening imports and changes.
        import os

        if os.environ.get("FINANCE_PAIRED_VALIDATION_PROFILE"):
            from src.agents.paired_validation import (
                install_from_environment as install_paired_validation,
            )

            install_paired_validation()

        if os.environ.get("FINANCE_SCREENING_ARM"):
            from src.agents.screening_runtime import install_from_environment as install_screening

            install_screening()
            if os.environ.get("FINANCE_SCREENING_RESOURCE_PROFILE"):
                from src.agents.screening_placement import (
                    install_from_environment as install_placement,
                )
                from src.agents.screening_reference import (
                    install_from_environment as install_reference,
                )

                install_reference()
                install_placement()
            if os.environ.get("FINANCE_SCREENING_ARTIFACT_PROFILE") == "actor_bf16_hf_v1":
                from src.agents.actor_export import install_worker_export_hook

                install_worker_export_hook()
            if os.environ.get("FINANCE_CRITIC_FIT_PROBE_DIR"):
                from src.agents.critic_fit_probe import (
                    install_from_environment as install_critic_fit,
                )

                install_critic_fit()
            if os.environ.get("FINANCE_ACTOR_UPDATE_PROBE_DIR"):
                from src.agents.actor_update_probe import (
                    install_from_environment as install_actor_update,
                )

                install_actor_update()
    except Exception as exc:
        # CPython deliberately swallows ordinary ``sitecustomize`` exceptions.
        # SystemExit is outside that handler and keeps required G1 hooks fail-closed.
        raise SystemExit(f"finance runtime hook installation failed: {exc}") from exc


_install_runtime_hooks()
