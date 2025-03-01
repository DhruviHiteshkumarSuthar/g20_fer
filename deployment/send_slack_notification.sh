#!/bin/bash

# Read the message and notification type from the parameters
SOURCE="$1"
NOTIFICATION_TYPE="$2"
TITLE="$3"
MESSAGE="$4"
DETAILS="$5"
ICON_EMOJI="$6"

# Set the emoji and color based on notification type
if [ "$NOTIFICATION_TYPE" == "" ]; then
    COLOR="#0000FF"
elif [ "$NOTIFICATION_TYPE" == "success" ]; then
    COLOR="good"
elif [ "$NOTIFICATION_TYPE" == "failure" ]; then
    COLOR="danger"
else
    ICON_EMOJI=":warning:"
    COLOR="warning"
fi

# Send the message to Slack
curl -X POST -H 'Content-type: application/json' --data "{
  \"text\": \"$ICON_EMOJI *$TITLE* \n$MESSAGE\nSource: $SOURCE \",
  \"attachments\": [{
    \"color\": \"$COLOR\",
    \"text\": \"$DETAILS\"
  }]
}" ${{ secrets.SLACK_URL }}
