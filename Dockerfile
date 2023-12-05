# Usar una imagen base de Python
FROM python:3.8-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo requirements.txt al contenedor
COPY requirements.txt /app/

# La API
ENV DW_AUTH_TOKEN=eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJwcm9kLXVzZXItY2xpZW50OmNhbGxtZW1ha2VyIiwiaXNzIjoiYWdlbnQ6Y2FsbG1lbWFrZXI6OjdjZWMyZDlhLWVhNGItNDg3Zi1iNjc2LTFiNjI4YTcyYWQ4YyIsImlhdCI6MTY5OTEwNjYxNywicm9sZSI6WyJ1c2VyX2FwaV9yZWFkIiwidXNlcl9hcGlfd3JpdGUiXSwiZ2VuZXJhbC1wdXJwb3NlIjp0cnVlLCJzYW1sIjp7fX0._A_n5T_I1zkrz1zDOOGdhPoEn7HeBt_J3fvloShHaMHw8Q4CmYcPKTwlh7fQtDGqD9QpE4o6wHD609zGhWBu0w

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar los archivos necesarios al contenedor
COPY . /app

# Exponer el puerto en el que Flask se ejecutará
EXPOSE 5000

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]
