//var http = require('http');
//
//http.createServer(function(request, response) {
//  var headers = request.headers;
//  var method = request.method;
//  var url = request.url;
//  var body = [];
//  request.on('error', function(err) {
//    console.error(err);
//  }).on('data', function(chunk) {
//    body.push(chunk);
//  }).on('end', function() {
//    body = Buffer.concat(body).toString();
//    // At this point, we have the headers, method, url and body, and can now
//    // do whatever we need to in order to respond to this request.
//    response.end('<html><body><h1>Hello, World!</h1></body></html>');
//  });
//}).listen(8080, '172.31.33.116'); // Activates this server, listening on port 8080.

var http = require('http'),
    inspect = require('util').inspect,
    path = require('path'),
    os = require('os'),
    fs = require('fs');
var exec = require('child_process').exec;

var Busboy = require('busboy');
process.env.CUDA_VISIBLE_DEVICES = ""
http.createServer(function(req, res) {
  if (req.method === 'POST') {
    var busboy = new Busboy({ headers: req.headers });
    busboy.on('file', function(fieldname, file, filename, encoding, mimetype) {
	var saveTo = path.join(os.tmpDir(), path.basename(filename));
	console.log(saveTo)
      file.pipe(fs.createWriteStream(saveTo));
	var spawn = require('child_process').spawn;
	var child = spawn('python', [
	  '../writeCaptions.py', '0.75',
	  './weights.h5', saveTo
	]);
	var output  = [];
	child.stdout.on('data', function(chunk) {
	  // output will be here in chunks
  	  console.log(chunk)
          output.push(chunk);
	});

	child.on('close', function(code) {
    		if (code === 0)
      			res.end(Buffer.concat(output));
    		else
      			res.end(500); // when the script fails, generate a Server Error HTTP response
  	});
    });
    //busboy.on('finish', function() {
    //  console.log('Done parsing form!');
    //  res.writeHead(200, { Connection: 'close', Location: '/' });
    //  res.end();
    //});
    req.pipe(busboy);
  } else if (req.method === 'GET') {
    res.writeHead(200, { Connection: 'close' });
    res.end('<html><head></head><body>\
               <form method="POST" enctype="multipart/form-data">\
                <input type="file" name="filefield"><br />\
                <input type="submit">\
              </form>\
            </body></html>');
  }
}).listen(8080, 'localhost', function() {
  console.log('Listening for requests');
});
