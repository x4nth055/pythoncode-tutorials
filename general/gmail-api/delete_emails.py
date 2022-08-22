from common import gmail_authenticate, search_messages

def delete_messages(service, query):
    messages_to_delete = search_messages(service, query)
    print(f"Deleting {len(messages_to_delete)} emails.")
    # it's possible to delete a single message with the delete API, like this:
    # service.users().messages().delete(userId='me', id=msg['id'])
    # but it's also possible to delete all the selected messages with one query, batchDelete
    return service.users().messages().batchDelete(
      userId='me',
      body={
          'ids': [ msg['id'] for msg in messages_to_delete]
      }
    ).execute()

if __name__ == "__main__":
    import sys
    service = gmail_authenticate()
    delete_messages(service, sys.argv[1])