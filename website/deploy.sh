#!/bin/bash

# Deploy script for GitHub Pages
echo "Building Text2Scene website..."

# Install dependencies
npm install

# Build the project
npm run build

echo "Build complete! The 'build' folder is ready for deployment to GitHub Pages."
echo "To deploy:"
echo "1. Copy contents of 'build' folder to your GitHub Pages repository"
echo "2. Or use gh-pages package: npm install --save-dev gh-pages"
echo "3. Add to package.json scripts: \"deploy\": \"gh-pages -d build\""
echo "4. Then run: npm run deploy"