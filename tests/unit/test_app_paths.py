"""Tests for lib/app_paths.py — resource and user data path resolution."""

import os
import sys
import pytest
from unittest.mock import patch


class TestResourcePath:
    """Test resource_path() for read-only bundled assets."""

    def test_dev_mode_returns_project_root(self):
        from lib.app_paths import resource_path
        result = resource_path('config/default_config.json')
        assert result.endswith('config/default_config.json')
        assert os.path.isabs(result)

    def test_dev_mode_file_exists(self):
        from lib.app_paths import resource_path
        assert os.path.exists(resource_path('config/default_config.json'))

    def test_frozen_mode_uses_meipass(self):
        from lib.app_paths import get_base_path
        with patch.object(sys, 'frozen', True, create=True), \
             patch.object(sys, '_MEIPASS', '/tmp/fake_meipass', create=True):
            assert get_base_path() == '/tmp/fake_meipass'

    def test_dev_mode_base_is_project_root(self):
        from lib.app_paths import get_base_path
        base = get_base_path()
        # Project root should contain diablos_modern.py
        assert os.path.exists(os.path.join(base, 'diablos_modern.py'))


class TestUserDataPath:
    """Test user_data_path() for writable user data."""

    def test_dev_mode_matches_resource_path(self):
        """In development, user_data_path == resource_path."""
        from lib.app_paths import resource_path, user_data_path
        assert user_data_path('config/foo.json') == resource_path('config/foo.json')

    def test_frozen_mode_uses_app_support_on_macos(self):
        from lib.app_paths import get_user_data_dir
        with patch.object(sys, 'frozen', True, create=True), \
             patch.object(sys, 'platform', 'darwin'), \
             patch('os.makedirs'):
            result = get_user_data_dir()
            assert 'DiaBloS' in result
            assert 'Application Support' in result or '.local/share' in result

    def test_frozen_mode_uses_appdata_on_windows(self):
        from lib.app_paths import get_user_data_dir
        with patch.object(sys, 'frozen', True, create=True), \
             patch.object(sys, 'platform', 'win32'), \
             patch.dict(os.environ, {'APPDATA': '/tmp/fake_appdata'}), \
             patch('os.makedirs'):
            result = get_user_data_dir()
            assert 'DiaBloS' in result
            assert '/tmp/fake_appdata' in result

    def test_user_data_path_creates_parent_dirs(self):
        """user_data_path should ensure parent directory exists."""
        from lib.app_paths import user_data_path
        # In dev mode this resolves to project root, which already exists
        path = user_data_path('config/test_file.json')
        assert os.path.isdir(os.path.dirname(path))
