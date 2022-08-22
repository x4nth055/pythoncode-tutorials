from common import gmail_authenticate, search_messages

def mark_as_read(service, query):
    messages_to_mark = search_messages(service, query)
    print(f"Matched emails: {len(messages_to_mark)}")
    return service.users().messages().batchModify(
      userId='me',
      body={
          'ids': [ msg['id'] for msg in messages_to_mark ],
          'removeLabelIds': ['UNREAD']
      }
    ).execute()

def mark_as_unread(service, query):
    messages_to_mark = search_messages(service, query)
    print(f"Matched emails: {len(messages_to_mark)}")
    return service.users().messages().batchModify(
        userId='me',
        body={
            'ids': [ msg['id'] for msg in messages_to_mark ],
            'addLabelIds': ['UNREAD']
        }
    ).execute()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Marks a set of emails as read or unread")
    parser.add_argument('query', help='a search query that selects emails to mark')
    parser.add_argument("-r", "--read", action="store_true", help='Whether to mark the message as read')
    parser.add_argument("-u", "--unread", action="store_true", help='Whether to mark the message as unread')

    args = parser.parse_args()
    service = gmail_authenticate()
    if args.read:
        mark_as_read(service, args.query)
    elif args.unread:
        mark_as_unread(service, args.query)
