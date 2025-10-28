export MYCHAT_BASE_DIR="$HOME/.cache/mychat"
mkdir -p $MYCHAT_BASE_DIR


# #download the eval bundle to evaluate Core metric when training
# EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
# if [ ! -d "$MYCHAT_BASE_DIR/eval_bundle" ]; then
#     curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
#     unzip -q eval_bundle.zip
#     rm eval_bundle.zip
#     mv eval_bundle $MYCHAT_BASE_DIR
# fi

# #download identify conversations to impart a personality to the model
# if [ ! -f "$MYCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
#     echo "Downloading identity conversations..."
#     curl -L -o "$MYCHAT_BASE_DIR/identity_conversations.jsonl" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
# fi