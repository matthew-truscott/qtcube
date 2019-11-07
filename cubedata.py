import uuid
from field import Field
from enum import Enum

class FileState(Enum):
    UNLOADED = 0
    ACTIVE = 1
    BYTES = 2

class CubeData():
    def __init__():
        # TODO consider a database to store
        # self.instance is a dictionary that contains all the cube information
        # KEY: unique ID
        # VAL: a dictionary of the following...
        # > "FILE": points to an absolute location on disk where the file is read from, can be loaded and unloaded (slow)
        # > "STATE": where is the data currently stored?
        # > "DATA": field object with all the info
        self.instance = {}

    def create_entry(filename):
        entryid = str(uuid.uuid1)
        self.instance[entryid] = {}
        self.instance[entryid]["FILE"] = filename
        self.instance[entryid]["STATE"] = FileState.UNLOADED
        self.instance[entryid]["DATA"] = Field()
        return entryid

    def _find_entry_by_filename(filename):
        entrylist = []
        for key in self.instance:
            if self.instance[key]["FILE"] == filename:
                entrylist.append(key)
        return key

    def load_entry(id):
        filename = self.instance[entryid]["FILE"]
        # move this until we know whether read is successful
        filename = self.instance[entryid]["STATE"] = FileState.ACTIVE
        filename = self.instance[entryid]["DATA"].read_gc(filename)  