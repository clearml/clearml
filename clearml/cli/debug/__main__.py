import argparse
import sys
from argparse import ArgumentParser
from pprint import pprint

from clearml import Task
from clearml.backend_api.session import Session
from clearml.config import ConfigWrapper
from clearml.utilities.pyhocon import HOCONConverter
from clearml.utilities.pyhocon.exceptions import ConfigMissingException
from clearml.version import __version__

verbose = 0


def print_(msg: str, verbosity: int = 0) -> None:
    if verbose <= 0:
        return
    if verbosity and verbose < verbosity:
        return
    print(msg)


def do_dump(args: argparse.Namespace) -> None:
    print_("Connecting to ClearML Server at {}".format(Session.get_api_server_host(config=ConfigWrapper._init())))

    session = Task._get_default_session()

    print_(
        "Server version {} ({} feature set), API version {}".format(
            session.server_version, Session.feature_set, Session.api_version
        )
    )

    msg = "Configuration dump{}:".format("" if not args.path else " ({})".format(args.path))
    print_(msg)
    print_("=" * len(msg))

    config = session.config._config._config
    prefix = ""

    if args.path:
        try:
            config = config.get(args.path)
            prefix = args.path
        except ConfigMissingException:
            raise ValueError("Request path {} cannot be found in configuration", format(args.path))

    if args.format.lower() == "dict":
        pprint(config.as_plain_ordered_dict(), indent=args.indent, width=120)
    elif args.format.lower() == "json":
        print(HOCONConverter.to_json(config, indent=args.indent))
    elif args.format.lower() == "yaml":
        print(prefix + ": " + HOCONConverter.to_yaml(config, indent=args.indent, level=1 if prefix else 0))
    elif args.format.lower() == "hocon":
        print(prefix + " " + HOCONConverter.to_hocon(config, indent=args.indent, level=1 if prefix else 0))


def do_token(_: argparse.Namespace) -> None:
    session = Task._get_default_session()
    decoded_token = session.get_decoded_token(session.token)
    pprint(decoded_token)


def setup_parser(parser: ArgumentParser) -> None:
    parser.add_argument("-v", "--verbose", action="count", default=0)

    commands = parser.add_subparsers(help="Debug actions", dest="command")

    token = commands.add_parser("token", help="Print token details")
    token.set_defaults(func=do_token)

    config = commands.add_parser("config", help="Configuration related commands")
    config_commands = config.add_subparsers(help="Config actions", dest="config_commands")
    dump = config_commands.add_parser("dump", help="Print configuration dump")

    dump.add_argument(
        "--format",
        "-F",
        choices=["json", "yaml", "dict", "hocon"],
        help="Output format (default %(default)s)",
        default="hocon",
    )
    dump.add_argument("--indent", "-I", default=2, type=int, help="Indentation (default: %(default)d)")
    dump.add_argument(
        "--path",
        "-p",
        type=str,
        help='Configuration path to dump (e.g. "api" or "sdk.aws.s3")',
    )

    dump.set_defaults(func=do_dump)


def cli() -> None:
    title = "ClearML Debug - debugging tools for using ClearML SDK"
    parser = ArgumentParser(description=title)
    setup_parser(parser)

    # get the args
    args = parser.parse_args()

    global verbose
    verbose = args.verbose

    print_("ClearML Version {}".format(__version__))

    if len(sys.argv) < 2:
        parser.print_help()
        exit(0)

    args.func(args)


def main() -> None:
    try:
        cli()
    except KeyboardInterrupt:
        print("\nUser aborted")
    except Exception as ex:
        print("\nError: {}".format(ex))
        exit(1)


if __name__ == "__main__":
    main()
