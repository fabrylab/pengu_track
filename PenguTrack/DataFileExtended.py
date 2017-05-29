import peewee
import clickpoints

class DataFileExtended(clickpoints.DataFile):

    def __init__(self, *args, **kwargs):
        clickpoints.DataFile.__init__(self, *args, **kwargs)

        class Measurement(self.base_model):
            # full definition here - no need to use migrate
            marker = peewee.ForeignKeyField(self.table_marker, unique=True, related_name="measurement",
                                            on_delete='CASCADE')  # reference to frame and track via marker!
            log = peewee.FloatField(default=0)
            x = peewee.FloatField()
            y = peewee.FloatField()

        if "measurement" not in self.db.get_tables():
            Measurement.create_table()  # important to respect unique constraint

        self.table_measurement = Measurement  # for consistency

    def setMeasurement(self, marker=None, log=None, x=None, y=None):
        assert not (marker is None), "Measurement must refer to a marker."
        try:
            item = self.table_measurement.get(marker=marker)
        except peewee.DoesNotExist:
            item = self.table_measurement()

        dictionary = dict(marker=marker, log=log, x=x, y=y)
        for key in dictionary:
            if dictionary[key] is not None:
                setattr(item, key, dictionary[key])
        item.save()
        return item

