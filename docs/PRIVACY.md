# Privacy notes

The API processes uploaded image bytes in memory and does not persist uploads or write them to an application database. The browser sends the selected image to the server for inference. Do not deploy publicly without HTTPS, authentication/rate limiting, abuse controls, retention review and a privacy notice. Reverse proxies and hosting providers may have their own request logging.
