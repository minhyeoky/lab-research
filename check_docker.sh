#!/usr/bin/env bash
docker ps --format "{{.ID}}: {{.Command}}" --no-trunc