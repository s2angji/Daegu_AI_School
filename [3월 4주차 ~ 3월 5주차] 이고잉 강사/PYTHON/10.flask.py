from flask import Flask, request, redirect


app = Flask(__name__)

topics = [
  {"id":1, "title":"html", "body":"html is ...."},
  {"id":2, "title":"css", "body":"css is ...."},
  {"id":3, "title":"js", "body":"js is ...."}
]
nextId = 4

def template(content, id=None):
  liTags = ''
  for topic in topics:
    liTags = liTags + f'<li><a href="/read/{topic["id"]}/">{topic["title"]}</a></li>'
  return f'''
  <html>
    <body>
      <h1><a href="/">WEB</a></h1>
      <ol>
        {liTags}
      </ol>
      {content}
      <ul>
        <li><a href="/create/">create</a></li>
        <li>
          <form action="/delete/{id}/" method="POST">
            <input type="submit" value="delete">
          </form>
        </li>
      </ul>
    </body>
  </html>
  '''

@app.route("/")
def index():
  return template('<h2>Welcome</h2>Hello, WEB!')

@app.route("/read/<int:id>/")
def read(id):
  title = ''
  body = ''  
  for topic in topics :
    if topic['id'] == id:
      title = topic['title']
      body = topic['body']
      break;
  return template(f'<h2>{title}</h2>{body}', id)

@app.route('/create/')
def create():
  content = '''
    <form action="/create_process/" method="POST">
      <p><input type="text" name="title" placeholder="title"></p>
      <p><textarea name="body" placeholder="body"></textarea></p>
      <p><input type="submit" value="create"></p>
    </form>
  '''
  return template(content)

@app.route('/create_process/', methods=['POST'])
def create_process():
  global nextId
  title = request.form['title']
  body = request.form['body']
  newTopic = {"id":nextId, "title": title, "body": body}
  topics.append(newTopic)
  nextId = nextId + 1
  return redirect(f'/read/{nextId-1}/')


@app.route('/delete/<int:id>/', methods=['POST'])
def delete(id):
  for topic in topics:
    if topic['id'] == id:
      topics.remove(topic)
      break;
  return redirect('/')

# @app.route('/update/')
# def update():
#   return 'Update'


app.run()