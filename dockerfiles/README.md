# System construction



## Use dockerfile to build systems
1.Use the dockerfile example we provided after modification or customize Dockerfile by yourself.

2.Using the command line: run the following command:

```
docker build -t {image_name} .
docker run -it --name {container_name} -p {Container_port}:{native port} -d {image_name}
```

3.Use port mapping to access docker or execute the following commands to access through bash:

```
docker exec -it {container_name} /bin/bash
```

**Note**: The dependency and deployment method of different versions of the software system may change. Please confirm the relevant information of the current version before deployment. If you use the dockerfile we provide, you need to modify the download link in dockerfile to adapt to different versions. The following is a common archive version download address:

- mysql: https://downloads.mysql.com/archives/
- mariadb: https://mariadb.org/mariadb/all-releases/
- apache: https://archive.apache.org/dist/httpd/
- gcc: https://www.gnu.org/prep/ftp.html
- clang: https://releases.llvm.org/download.html

