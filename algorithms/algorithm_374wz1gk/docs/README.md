# Left and Right Turn algorithms for self drive

In order to run the algorithm you must

```
left_obj = leftTurn()
left_obj.yellow_mask = dict["yellow"]
left_obj.final = combined
left_obj.white_mask = dict["white"]
left_obj.state_machine()
waypoints = left_obj.find_center_of_lane()
```

`find_center_of_lane()` will return a list of waypoints that the robot should travel to to reach the goal. 

For reference this is how to then get a waypoint into occupancy grid frame.

```
waypoint = np.array([x, y, 1])
multiplied_waypoint = self.zed.matrix @ waypoint
multiplied_waypoint /= multiplied_waypoint[2]
tx, ty = multiplied_waypoint[0], multiplied_waypoint[1]
tx *= self.scale_factor
ty *= self.scale_factor

self.waypoint_in_frame = (tx, ty)
```