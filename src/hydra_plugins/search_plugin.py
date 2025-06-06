from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class CustomSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        """Appends the search path for this plugin to the end of the hydra search path.
        This makes all config files inside the conf folder available by their group/name.
        """
        import qfat.conf

        qfat.conf.register_all()

        search_path.append(provider="qfat", path="pkg://qfat/conf")
