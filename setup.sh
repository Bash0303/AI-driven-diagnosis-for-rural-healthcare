#!/bin/bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = \$PORT\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
[browser]\n\
serverAddress = \"0.0.0.0\"\n\
serverPort = \$PORT\n\
" > ~/.streamlit/config.toml