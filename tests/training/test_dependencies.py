from talongym.training import dependencies


def test_install_command_uses_the_active_interpreter_and_rl_extra():
    command = dependencies.install_command("rl")
    assert "-m pip install -e \".[rl]\"" in command
    assert ".venv/bin/python" not in command


def test_install_command_quotes_a_windows_interpreter_path(monkeypatch):
    monkeypatch.setattr(dependencies.sys, "platform", "win32")
    monkeypatch.setattr(dependencies.sys, "executable", r"C:\Users\Team Member\Python\python.exe")
    assert dependencies.install_command("rl") == (
        '"C:\\Users\\Team Member\\Python\\python.exe" -m pip install -e ".[rl]"'
    )


def test_grpo_missing_dependency_error_uses_the_portable_install_command(monkeypatch):
    import builtins

    from talongym.presets.loader import load_bundle
    from talongym.training.grpo import train_grpo

    original_import = builtins.__import__

    def import_without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    try:
        train_grpo(bundle=load_bundle())
    except RuntimeError as exc:
        assert ".[rl]" in str(exc)
        assert ".venv/bin/python" not in str(exc)
    else:
        raise AssertionError("GRPO should require the RL extra when torch is unavailable")
