# [How to Generate and Read QR Code in Python](https://www.thepythoncode.com/article/generate-read-qr-code-python)
To run this:
- `pip3 install -r requirements.txt`
- If you want to generate a QR code, run `generate_qrcode.py`.

    For instance, if you want to generate a QR code that contains the data: "https://www.thepythoncode.com" to a file named `site.png`, you can:
    ```
    python generate_qrcode.py https://www.thepythoncode.com site.png
    ```
- If you want to read a QR code, run `read_qrcode.py`.

    For instance, if you want to read a QR code from a file named `site.png`, you can run:
    ```
    python read_qrcode.py site.png
    ```
    A new window will appear that contains the QR code surrounded by a blue square.
    and **outputs**:
    ```
    QRCode data:
    https://www.thepythoncode.com
    ```
- If you want to read QR codes live using your cam, just run:
    ```
    python read_qrcode_live.py
    ```

If you want to know how these are created, head to this tutorial: [How to Generate and Read QR Code in Python](https://www.thepythoncode.com/article/generate-read-qr-code-python).