FROM node:20-alpine

WORKDIR /app

COPY package.json ./
RUN npm install

COPY . .

RUN npm run build

EXPOSE 3000

ENV BACKEND_URL=http://backend:8000

CMD ["npm", "start"]
