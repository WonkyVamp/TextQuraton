const http = require("https");

const options = {
    "method": "POST",
    "hostname": "pen-to-print-handwriting-ocr.p.rapidapi.com",
    "port": null,
    "path": "/recognize/",
    "headers": {
        "content-type": "multipart/form-data; boundary=---011000010111000001101001",
        "X-RapidAPI-Key": "1f217dbc14mshfe0cd3cbcf7fef6p1c6587jsna8a47ef9543e",
        "X-RapidAPI-Host": "pen-to-print-handwriting-ocr.p.rapidapi.com",
        "useQueryString": true
    }
};

const req = http.request(options, function (res) {
    const chunks = [];

    res.on("data", function (chunk) {
        chunks.push(chunk);
    });

    res.on("end", function () {
        const body = Buffer.concat(chunks);
        console.log(body.toString());
    });
});

req.write(`-----011000010111000001101001\r
Content - Disposition: form - data; name =\"srcImg\"\r
\r
\r
----- 011000010111000001101001\r
Content - Disposition: form - data; name =\"Session\"\r
\r
string\r
----- 011000010111000001101001--\r
\r
`);
req.end();