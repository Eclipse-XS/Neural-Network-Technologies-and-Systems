"""Lifecycle fault injection; these tests do not claim live Docker/LM coverage."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('startup', Path(__file__).resolve().parents[1] / 'scripts/startup.py')
startup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(startup)


class StartupTests(unittest.TestCase):
    def test_poll_requires_positive_readiness(self):
        check = Mock(side_effect=[False, OSError('not ready'), True])
        with patch.object(startup.time, 'sleep'):
            startup.wait_ready(check, 'service')
        self.assertEqual(check.call_count, 3)

    def test_failed_child_aborts_wait(self):
        child = Mock(returncode=1)
        child.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, 'Streamlit exited'):
            startup.wait_ready(lambda: True, 'UI', process=child)

    def test_windows_job_kills_owned_process(self):
        if sys.platform != 'win32':
            self.skipTest('Windows only')
        job = startup.ChildJob()
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        try:
            job.assign(child)
            job.close()
            child.wait(timeout=10)
            self.assertIsNotNone(child.returncode)
        finally:
            if child.poll() is None:
                child.kill(); child.wait()

    def scenario(self, *, docker=True, redis_running=True, llm=True, port_busy=False, stop=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            redis = Mock(redis_url='redis://localhost:6380')
            redis.ping.side_effect = [redis_running, True]
            probe = types.SimpleNamespace(is_ready=llm, model_id='test-model', status=types.SimpleNamespace(value='NO_MODEL_LOADED'), detail='No model loaded')
            modules = {
                'knowledge_copilot.config': types.SimpleNamespace(LAB_ROOT=root),
                'knowledge_copilot.api.dependencies': types.SimpleNamespace(get_redis_adapter=lambda: redis, get_sqlite_repo=Mock()),
                'knowledge_copilot.infrastructure.llm_client': types.SimpleNamespace(probe_lm_studio_readiness=Mock(return_value=probe)),
                'knowledge_copilot.bootstrap': types.SimpleNamespace(initialize_knowledge=Mock()),
            }
            child = Mock(returncode=0)
            child.poll.return_value = None
            if stop:
                child.wait.side_effect = [KeyboardInterrupt, 0]
            sock = Mock()
            if port_busy:
                sock.bind.side_effect = OSError('busy')
            socket_context = Mock()
            socket_context.__enter__ = Mock(return_value=sock)
            socket_context.__exit__ = Mock(return_value=False)
            commands = Mock(side_effect=lambda args, **kw: '' if '--status' in args else 'container-123')
            cwd = Path.cwd()
            try:
                with patch.dict(sys.modules, modules), patch.object(startup, 'ROOT', root), patch.object(startup, 'load_environment'), patch.object(startup, 'dependencies'), patch.object(startup, 'command', commands), patch.object(startup, 'ChildJob'), patch.object(startup, 'wait_ready'), patch.object(startup.socket, 'socket', return_value=socket_context), patch.object(startup.shutil, 'which', return_value='docker' if docker else None), patch.object(startup.subprocess, 'Popen', return_value=child), patch.object(sys, 'argv', ['startup.py', '--no-browser']):
                    error = None
                    try:
                        startup.main()
                    except (RuntimeError, KeyboardInterrupt) as exc:
                        error = exc
            finally:
                startup.os.chdir(cwd)
            return commands, child, error

    def test_docker_missing(self):
        commands, child, error = self.scenario(docker=False)
        self.assertIn('Docker CLI missing', str(error))
        commands.assert_not_called()

    def test_no_model_does_not_stop_existing_redis(self):
        commands, child, error = self.scenario(llm=False)
        self.assertIn('NO_MODEL_LOADED', str(error))
        self.assertFalse(any(c.args[0][:2] == ['docker', 'stop'] for c in commands.call_args_list))

    def test_no_model_stops_only_owned_redis(self):
        commands, child, error = self.scenario(llm=False, redis_running=False)
        self.assertIn('NO_MODEL_LOADED', str(error))
        self.assertTrue(any(c.args[0] == ['docker', 'stop', 'container-123'] for c in commands.call_args_list))

    def test_port_conflict_does_not_start_services(self):
        commands, child, error = self.scenario(port_busy=True)
        self.assertIn('8501 is occupied', str(error))
        commands.assert_not_called()

    def test_interrupt_terminates_owned_ui_preserves_existing_redis(self):
        commands, child, error = self.scenario(stop=True)
        self.assertIsInstance(error, KeyboardInterrupt)
        if sys.platform != 'win32':
            child.terminate.assert_called_once()
        self.assertFalse(any(c.args[0][:2] == ['docker', 'stop'] for c in commands.call_args_list))


if __name__ == '__main__':
    unittest.main()
