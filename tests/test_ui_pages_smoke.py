import importlib
import unittest


class UITests(unittest.TestCase):
    def test_every_page_module_imports_and_has_render_page(self):
        pages = [
            'ui.pages.dashboard',
            'ui.pages.live_monitoring',
            'ui.pages.analytics',
            'ui.pages.history',
            'ui.pages.evaluation',
            'ui.pages.dataset',
            'ui.pages.training',
            'ui.pages.models',
            'ui.pages.alerts',
            'ui.pages.system_health',
            'ui.pages.settings',
            'ui.pages.user_management',
            'ui.pages.placeholder',
        ]
        for mod_name in pages:
            mod = importlib.import_module(mod_name)
            self.assertTrue(hasattr(mod, 'render_page'))


if __name__ == '__main__':
    unittest.main()

