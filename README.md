* here is how to use git with different credentials. Apply this command before pushing
```
git config --local credential.helper ""
git config --local user.name 'netphar'
git config --local user.email forfimm2018@gmail.com
git config --local credential.username netphar
```
* here is how to split big files
```
to untar `cat big_data_with_Loewe.tar.gz.parta* | tar -xvf -`
to tar `tar -zcvf datalist_with_Loewe.tar.gz datalist_with_Loewe`
to split `split -b 90M big_data_with_Loewe.tar.gz :`
```