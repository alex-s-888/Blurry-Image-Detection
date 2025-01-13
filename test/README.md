## Batch job

Batch prediction job may be executed using docker image `my_docker_image` created earlier (see [deployment_docker/README.md](../deployment_docker/README.md)).  
You will have to mount the input/output folders and pass input and output as parameters to docker, e.g:  
`docker run --rm -ti -v {input_folder}:/app/input/ -v {output_folder}:/app/output/ my_docker_image ./input ./output/{output_file_name}` 

In my case the actuall command was the following:  
`docker run --rm -ti -v c:/_temp/input/:/app/input/ -v c:/_temp/output/:/app/output/ my_docker_image ./input ./output/prediction.csv`
