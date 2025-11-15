# NOTE: NEED TO HAVE YOUR AWS CLI SET UP AND CONFIGURED TO RUN THIS SCRIPT
# CD INTO YOUR DATA SET DIRECTORY FIRST AND THEN RUN THIS SCRIPT
#!/usr/bin/env bash

# ==== CONFIG ====
# REQUIRED: set your bucket name once (or pass BUCKET=... on the command line)
: "${BUCKET:=visight-data-yusufmoola}"

# S3 prefix under the bucket for this dataset version
: "${PREFIX:=raw/roboflow/v8}"

# ==== HELPERS ====
aws_cmd() {
  if [[ -n "$AWS_PROFILE" ]]; then
    aws --profile "$AWS_PROFILE" "$@"
  else
    aws "$@"
  fi
}

# ==== PRECHECKS ====
echo "Bucket: s3://$BUCKET"
echo "Prefix: s3://$BUCKET/$PREFIX/"
[[ -f "data.yaml" ]] || { echo "Error: data.yaml not found in current directory."; exit 1; }
[[ -d "train" ]]    || { echo "Error: train/ folder not found."; exit 1; }
[[ -d "valid" ]]    || { echo "Error: valid/ folder not found."; exit 1; }
[[ -d "test" ]]     || { echo "Error: test/ folder not found."; exit 1; }

# Make sure the top-level prefixes exist (not strictly required, but keeps things tidy)
aws_cmd s3api put-object --bucket "$BUCKET" --key "$PREFIX/" || true
aws_cmd s3api put-object --bucket "$BUCKET" --key "$PREFIX/train/images/" || true
aws_cmd s3api put-object --bucket "$BUCKET" --key "$PREFIX/train/labels/" || true
aws_cmd s3api put-object --bucket "$BUCKET" --key "$PREFIX/valid/images/" || true
aws_cmd s3api put-object --bucket "$BUCKET" --key "$PREFIX/valid/labels/" || true
aws_cmd s3api put-object --bucket "$BUCKET" --key "$PREFIX/test/images/"  || true
aws_cmd s3api put-object --bucket "$BUCKET" --key "$PREFIX/test/labels/"  || true

echo "Uploading data.yaml..."
aws_cmd s3 cp ./data.yaml "s3://$BUCKET/$PREFIX/data.yaml"

echo "Syncing train/ ..."
aws_cmd s3 sync ./train  "s3://$BUCKET/$PREFIX/train/"  --only-show-errors

echo "Syncing valid/ ..."
aws_cmd s3 sync ./valid  "s3://$BUCKET/$PREFIX/valid/"  --only-show-errors

echo "Syncing test/  ..."
aws_cmd s3 sync ./test   "s3://$BUCKET/$PREFIX/test/"   --only-show-errors

echo "Verifying upload..."
aws_cmd s3 ls "s3://$BUCKET/$PREFIX/" --recursive --human-readable --summarize

echo "Done!"