import dataclasses
from datetime import datetime
import requests

@dataclasses.dataclass
class Request:
    query: str = None
    frm: datetime = None
    to: datetime = None
    type: str = None
    offset: int = None
    siteSearch: str = None
    collection: str = None
    maxItems: int = None
    itemsPerSite: int = None
    dedupValue: int = None
    dedupField: str = None
    fields: str = None
    callback: str = None
    prettyPrint: bool = None

class Interface:

    def __init__(self):
        self._requests = 0

    def text_request(self, request: Request) -> dict:
        payload = {}
        payload['q'] = request.query
        if request.frm: payload['from'] = request.frm
        if request.to: payload['to'] = request.to
        if request.type: payload['type'] = request.type
        if request.offset: payload['offset'] = request.offset
        if request.siteSearch: payload['siteSearch'] = request.siteSearch
        if request.collection: payload['collection'] = request.collection
        payload['maxItems'] = request.maxItems if request.maxItems else 5
        if request.itemsPerSite: payload['itemsPerSite'] = request.itemsPerSite
        if request.dedupValue: payload['dedupValue'] = request.dedupValue
        if request.dedupField: payload['dedupField'] = request.dedupField
        if request.fields: payload['fields'] = request.fields
        if request.callback: payload['callback'] = request.callback
        if request.prettyPrint: payload['prettyPrint'] = request.prettyPrint
        r = requests.get('http://arquivo.pt/textsearch', params=payload)
        self._requests += 1
        return r.json()

    def retrieve_text_from_link(self, link: str):
        page = requests.get(link)
        return page.content.decode('utf-8')

if __name__ == '__main__':
    interface = Interface()
    request = Request(query='SNS')
    print("inicio do request")
    json = interface.text_request(request)
    print(json)
    for item in json["response_items"]:
        title = item["title"]
        url = item["linkToArchive"]
        time = item["tstamp"]

        print(title)
        print(url)
        print(time)

        content = interface.retrieve_text_from_link(item["linkToExtractedText"])

        print(content)
        print("\n")