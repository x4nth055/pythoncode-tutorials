# for parsing commandline arguments
import argparse
from common import search_messages, gmail_authenticate
from read_emails  import read_message
from send_emails  import send_message
from delete_emails import delete_messages
from mark_emails  import mark_as_read, mark_as_unread


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Send/Search/Delete/Mark messages using gmail's API.")
    subparsers = parser.add_subparsers(help='Subcommands')
    parser_1 = subparsers.add_parser('send', help='Send an email')
    parser_1.add_argument('destination', type=str, help='The destination email address')
    parser_1.add_argument('subject', type=str, help='The subject of the email')
    parser_1.add_argument('body', type=str, help='The body of the email')
    parser_1.add_argument('files', type=str, help='email attachments', nargs='+')
    parser_1.set_defaults(action='send')
    parser_2 = subparsers.add_parser('delete', help='Delete a set of emails')
    parser_2.add_argument('query', type=str, help='a search query that selects emails to delete')
    parser_2.set_defaults(action='delete')
    parser_3 = subparsers.add_parser('mark', help='Marks a set of emails as read or unread')
    parser_3.add_argument('query', type=str, help='a search query that selects emails to mark')
    parser_3.add_argument('read_status', type=bool, help='Whether to mark the message as unread, or as read')
    parser_3.set_defaults(action='mark')
    parser_4 = subparsers.add_parser('search', help='Marks a set of emails as read or unread')
    parser_4.add_argument('query', type=str, help='a search query, which messages to display')
    parser_4.set_defaults(action='search')
    args = parser.parse_args()
    service = gmail_authenticate()
    if args.action == 'send':
        # TODO: add attachements
        send_message(service, args.destination, args.subject, args.body, args.files)
    elif args.action == 'delete':
        delete_messages(service, args.query)
    elif args.action == 'mark':
        print(args.unread_status)
        if args.read_status:
            mark_as_read(service, args.query)
        else:
            mark_as_unread(service, args.query)
    elif args.action == 'search':
        results = search_messages(service, args.query)
        for msg in results:
            read_message(service, msg)
