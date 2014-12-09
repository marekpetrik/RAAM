#!/bin/bash
# A script to tag and automatically edit the version number in the version file

if [ "$#" -l 1 ]; then
    echo "Needs the tag name as a parameter."
    exit;
fi

echo "Creating tag '$1' ... Make sure the changes are commited."

while true; do
    read -p "Do you wish to create the tag?" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

echo "Updating version.py ..."
echo -e "# This file is updated automatically, do not edit.\nversion = '$1'" > raam/version.py

echo "Commiting updated version file ..."
git commit -m "Updated version file" raam/version.py 

echo "Creating tag ..."
git tag -a $1

echo "Pushing to origin ..."

git push --tags
