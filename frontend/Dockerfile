FROM node:20-alpine

WORKDIR /app

COPY package.json ./
RUN npm install

COPY . .

# Build with a placeholder - the actual URL is used at runtime via API routes
RUN npm run build

EXPOSE 3000

# This env var is read at RUNTIME by the API route handlers
ENV BACKEND_URL=http://backend:8000

CMD ["npm", "start"]
