from common import gmail_authenticate, search_messages

def mark_as_read(service, query):
    messages_to_mark = search_messages(service, query)
    if len(messages_to_mark) == 0: # No emails found
        return print("No emails found")
    else:
        print("="*50)
        for message_id in messages_to_mark:
            msg = service.users().messages().get(userId='me', id=message_id['id'], format='full').execute()
            payload = msg['payload']
            headers = payload.get("headers")
            if headers:
                # this section prints email basic info & creates a folder for the email
                for header in headers:
                    name = header.get("name")
                    value = header.get("value")
                    if name == 'From':
                        # we print the From address
                        print("From:", value)
                    if name == "To":
                        # we print the To address
                        print("To:", value)
                    if name == "Subject":
                        # we print the Subject
                        print("Subject:", value)
                    if name == "Date":
                        # we print the date when the message was sent
                        print("Date:", value)
            print("="*50)
            print("MARKED AS READ")
    return service.users().messages().batchModify(
      userId='me',
      body={
          'ids': [ msg['id'] for msg in messages_to_mark ],
          'removeLabelIds': ['UNREAD']
      }
    ).execute()

def mark_as_unread(service, query):
    messages_to_mark = search_messages(service, query)
    if len(messages_to_mark) == 0: # No emails found
        return print("No emails found")
    else:
        print("="*50)
        for message_id in messages_to_mark:
            msg = service.users().messages().get(userId='me', id=message_id['id'], format='full').execute()
            payload = msg['payload']
            headers = payload.get("headers")
            if headers:
                # this section prints email basic info & creates a folder for the email
                for header in headers:
                    name = header.get("name")
                    value = header.get("value")
                    if name == 'From':
                        # we print the From address
                        print("From:", value)
                    if name == "To":
                        # we print the To address
                        print("To:", value)
                    if name == "Subject":
                        # we print the Subject
                        print("Subject:", value)
                    if name == "Date":
                        # we print the date when the message was sent
                        print("Date:", value)
            print("="*50)
            print("MARKED AS UNREAD")
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

    service = gmail_authenticate()

    args = parser.parse_args()
    if args.read:
        mark_as_read(service, '"' + args.query + '" and label:unread' )
    elif args.unread:
        mark_as_unread(service, args.query)
