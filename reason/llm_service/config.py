from config.argument_parser import BaseArgs, FlexibleArgumentParser


class ServerArgs(BaseArgs):
    host: str = "0.0.0.0"
    controller_port: int = 28777

    @property
    def controller_addreress(self):
        return f"http://{self.host}:{self.controller_port}"

    def add_cli_args(parser: FlexibleArgumentParser):
        parser.add_argument("--host", type=str, default=ServerArgs.host)
        parser.add_argument(
            "--controller_port", type=int, default=ServerArgs.controller_port
        )
