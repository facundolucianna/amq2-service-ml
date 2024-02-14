# amq2-service-ml
 Estructura de servicios para la implementacion del proyecto final de AMq2

Para levantar el profible debug
docker compose --profile debug up

Para matar todo y borrar todo
 docker compose down --rmi all --volume 

En debug podemos usar el CLI de airflow

docker-compose run airflow-cli config list      


MacOS y Linux

Cambiar .env a AIRFLOW_UID por el UID del usuario

Para obtener el uid, usar el comando id, o 

sudo cat /etc/passwd | grep ${USER}