#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/OccupancyGrid.h>

#include <gridmap_2d/GridMap2D.h>

#include <message_filters/subscriber.h>


#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/UInt16.h>
#include <std_msgs/Bool.h>
#include <fstream>

// PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/vector_average.h>

#include <string>
#include <tf/tf.h>

const int class_colours[8][3] =
{
	{255  , 0, 0},
	{0, 255  , 0},
	{0  , 0, 255 },
	{100, 0  , 0  },
	{0  , 100  , 0  },
	{0  , 0  , 100  },
	{255  , 255  , 0  },
	{255  , 0  , 255  },
};

std::string class_names[8] =
{
"chair",
"table",
"window",
"door",
"box",
"shelves",
"sofa",
"cabinet"
};

/****** GLOBALS *************/
/* Publishers for pointclouds */
ros::Publisher pub_instance;
ros::Publisher pub_target;
ros::Publisher pub_goal_position;
/* Global variables with instance selection information */
uint16_t global_class_id = 0;
uint16_t global_instance_id = 0;
bool global_start_moving = false;
gridmap_2d::GridMap2D global_grid_map;

/* Number of classes */
const uint8_t num_classes = 8;

/* Function used to find the direction of the front of the chair */
bool find_chair_direction(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::ModelCoefficients::Ptr coefficients)
{
	double max_positive = 0;
	double max_negative = 0;

	for(int i=0; i<cloud->points.size(); i++)
	{
		double point_plane_dist;

		/* calculate the distance of point from plane (signed) */
		point_plane_dist = cloud->points[i].x * coefficients->values[0] +
				  		   cloud->points[i].y * coefficients->values[1] +
				  		   cloud->points[i].z * coefficients->values[2] +
				  					 		    coefficients->values[3]  ;

		/* Keep track of maximum distance in each direction */
		if(point_plane_dist > 0)
		{
			if(point_plane_dist>max_positive)
			{
				max_positive = point_plane_dist;
			}
		}
		else
		{
			if(-1*point_plane_dist>max_negative)
			{
				max_negative = -1*point_plane_dist;
			}
		}
	}

	/* Flip the plane if neccessary */
	if(max_negative>max_positive)
	{
		coefficients->values[0] = -1 * coefficients->values[0];
		coefficients->values[1] = -1 * coefficients->values[1];
		coefficients->values[2] = -1 * coefficients->values[2];
	}		

	/* Return if plane had to be flipped */
	return(max_negative>max_positive);
}

/* This function is used to check if given point is occupied */
bool check_if_valid(pcl::PointXYZ *outputPoint)
{
	bool isOccupied = global_grid_map.isOccupiedAt(outputPoint->x, outputPoint->y);
	bool isDistanceValid = (global_grid_map.distanceMapAt(outputPoint->x, outputPoint->y) >= 0.5);

	return((!isOccupied) && isDistanceValid);
}

/* This function is used to find a clear point on given vector */
bool find_point_on_ray(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, Eigen::Vector2f coefficients, Eigen::Vector3f mean_point, pcl::PointXYZ *outputPoint, float distance_range)
{
	bool pointFound = false;

	for(float current_distance = 0; current_distance<=distance_range; current_distance+=0.1)
	{
  		outputPoint->x = mean_point(0) + current_distance*coefficients(0);
  		outputPoint->y = mean_point(1) + current_distance*coefficients(1);

  		if(check_if_valid(outputPoint))
  		{
  			pointFound = true;
			break;
  		}
	}

	return(pointFound);
}

/* This function is used to find a clear point on given vector */
bool find_point_on_arc(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, Eigen::Vector2f coefficients, Eigen::Vector3f mean_point, pcl::PointXYZ *outputPoint, float angle_range, float current_distance)
{
	bool pointFound = false;
	Eigen::Vector2f rotated_coefficients;

	for(float current_angle = 0; current_angle<=angle_range; current_angle+=0.1)
	{
		rotated_coefficients(0) = coefficients(0)*cos(current_angle) - 
		                          coefficients(1)*sin(current_angle);
		rotated_coefficients(1) = coefficients(0)*sin(current_angle) + 
		                          coefficients(1)*cos(current_angle);			                      

  		outputPoint->x = mean_point(0) + current_distance*rotated_coefficients(0);
  		outputPoint->y = mean_point(1) + current_distance*rotated_coefficients(1);

  		if(check_if_valid(outputPoint))
  		{
  			pointFound = true;
			break;
  		}
  		
		rotated_coefficients(0) = coefficients(0)*cos(-1*current_angle) - 
		                          coefficients(1)*sin(-1*current_angle);
		rotated_coefficients(1) = coefficients(0)*sin(-1*current_angle) + 
		                          coefficients(1)*cos(-1*current_angle);			                      		                      

  		outputPoint->x = mean_point(0) + current_distance*rotated_coefficients(0);
  		outputPoint->y = mean_point(1) + current_distance*rotated_coefficients(1);

  		if(check_if_valid(outputPoint))
  		{
  			pointFound = true;
			break;
  		}
	}

	return(pointFound);
}

/* This function is used to find a clear point */
bool find_clear_point(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, Eigen::Vector2f coefficients, Eigen::Vector3f mean_point, pcl::PointXYZ *outputPoint, float angle_range, float distance_range)
{
	bool pointFound = false;
	Eigen::Vector2f rotated_coefficients;

	for(float current_angle = 0; current_angle<=angle_range; current_angle+=0.1)
	{
		if(current_angle == 0)
		{
			rotated_coefficients(0) = coefficients(0);
			rotated_coefficients(1) = coefficients(1);
			pointFound = find_point_on_ray(cloud, rotated_coefficients, mean_point, outputPoint, distance_range);
			if(pointFound)
			{
				break;
			}			
		}
		else
		{
			rotated_coefficients(0) = coefficients(0)*cos(current_angle) - 
			                          coefficients(1)*sin(current_angle);
			rotated_coefficients(1) = coefficients(0)*sin(current_angle) + 
			                          coefficients(1)*cos(current_angle);
			pointFound = find_point_on_ray(cloud, rotated_coefficients, mean_point, outputPoint, distance_range);
			if(pointFound)
			{
				break;
			}			

			rotated_coefficients(0) = coefficients(0)*cos(-1*current_angle) - 
			                          coefficients(1)*sin(-1*current_angle);
			rotated_coefficients(1) = coefficients(0)*sin(-1*current_angle) + 
			                          coefficients(1)*cos(-1*current_angle);
			pointFound = find_point_on_ray(cloud, rotated_coefficients, mean_point, outputPoint, distance_range);
			if(pointFound)
			{
				break;
			}	
		}
	}
	return pointFound;
}

bool find_clear_point_arc(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, Eigen::Vector2f coefficients, Eigen::Vector3f mean_point, pcl::PointXYZ *outputPoint, float angle_range, float distance_range)
{
	bool pointFound = false;

	for(float current_distance = 0; current_distance<=distance_range; current_distance+=0.1)
	{
		pointFound = find_point_on_arc(cloud, coefficients, mean_point, outputPoint, angle_range, current_distance);
		if(pointFound)
		{
			break;
		}			
	}
	return pointFound;
}


/* Main callback called when semantic pointcloud is available */
void callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
	/******** DECLARATIONS **********/
  	/* Create a ROS message to publish pointclouds */
    	sensor_msgs::PointCloud2 pcl_message;

	/* Matrix of vectors holding cluster extraction results */
	std::vector<pcl::PointIndices> cluster_indices[num_classes];

	/* Allocate memory for the PCL pointcloud */
  	pcl::PointCloud<pcl::PointXYZRGB>::Ptr pclCloud (new pcl::PointCloud<pcl::PointXYZRGB>);

  	/* KdTree object */
  	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);

  	/* Cluster extraction object */
  	pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> extractor;

	/* Create a vector with point clouds for each of the classes */
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_class[num_classes];
	for(int i =0; i<num_classes; i++)
	{
		cloud_class[i] = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);	
	}



	/* Save the header */
	std_msgs::Header header;
	header = cloud_msg->header;

  	/* Convert the recieved message to PCL pointcloud */
  	pcl::fromROSMsg(*cloud_msg, *pclCloud);

	/* Split the general cloud into class clouds */
  	for(int i=0; i<pclCloud->points.size(); i++)
  	{
  		for(int j = 0; j<num_classes; j++)
  		{
  			if((int)pclCloud->points[i].r == class_colours[j][0] && 
  			   (int)pclCloud->points[i].g == class_colours[j][1] &&
  			   (int)pclCloud->points[i].b == class_colours[j][2]    )
  			{
				cloud_class[j]->points.push_back(pclCloud->points[i]);
				break;
  			}
  		}
  	}

  	/* Display information */
  	ROS_INFO("Finished spliting cloud into class clouds");
  	ROS_INFO("Total points: %d", (int)pclCloud->points.size());

	for(int i =0; i<num_classes; i++)
	{
		ROS_INFO("Class %d cloud consists of %d points", i, (int)cloud_class[i]->points.size());
	}
  	



	/******* CLUSTER EXTRACTION ********/
  	
  	/* Set extraction parameters */
  	extractor.setClusterTolerance (0.1);
  	extractor.setMinClusterSize (50);
  	extractor.setMaxClusterSize (25000);  	

  	/* Extract clusters from each class pointcloud */
	for(int i =0; i<num_classes; i++)
	{
		/* If the class has no points assigned - continue */
		if(cloud_class[i]->points.size() == 0)
		{
			continue;
		}

		/* Create the tree */
  		tree->setInputCloud(cloud_class[i]);

  		/* Select pointcloud */
  		extractor.setSearchMethod(tree);
  		extractor.setInputCloud(cloud_class[i]);

  		/* Extract cluster indices */
  		extractor.extract(cluster_indices[i]);

  		ROS_INFO("For class %d extracted %d clusters", i, (int)cluster_indices[i].size());
	}  
  	



	/* 
	Go through all extracted clusters to:
			-calculate mean
			-save the pointcloud to file
	*/

	/* Declarations for this part */
  	pcl::VectorAverage<float, 3> vector_mean;
	pcl::PointXYZRGB newPoint;
	int instance_id = 0;
	std::ofstream myfile;
	pcl::PointIndices::Ptr one_side(new pcl::PointIndices());
  	pcl::ExtractIndices<pcl::PointXYZRGB> extract;

	/* Open results file */
	myfile.open ("/home/aut/s160948/results.txt", std::ofstream::out | std::ofstream::trunc);

	for(int class_id = 0; class_id<num_classes; class_id++)
	{
		/* Go through all of the instances for currently processed class */
		for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices[class_id].begin(); it != cluster_indices[class_id].end (); ++it)
	  	{
	  		/* Get the currently processed instance id */
	  		instance_id = (int)std::distance<std::vector<pcl::PointIndices>::const_iterator>(cluster_indices[class_id].begin(), it);

	  		/* Reset the vector mean data */
	  		vector_mean.reset();

	  		/* Create a cloud and assign a readom color for this instance */
	    	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);

	    	/* Loop through all of the indices for currently processed instance */
	    	for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end (); ++pit)
	    	{
	    		/* Create a new point with selected colour */
	    		newPoint.x = cloud_class[class_id]->points[*pit].x;
	    		newPoint.y = cloud_class[class_id]->points[*pit].y;
	    		newPoint.z = cloud_class[class_id]->points[*pit].z;
	    		newPoint.r = 255;
	    		newPoint.g = 255;
	    		newPoint.b = 255;
	    		/* Push the point the the cloud */
	      		cloud_cluster->points.push_back(newPoint);

	      		/* Add point to mean calculation */
	      		vector_mean.add(Eigen::Vector3f(newPoint.x, newPoint.y, newPoint.z));
	    	}

    		/* The cluster cloud has been created - calculate mean */
			Eigen::Vector3f mean_val = vector_mean.getMean();

			/* Print Information */
			myfile << "Instance " << instance_id << " of class " << class_names[class_id] << " consists of " << (int)cloud_cluster->points.size() << " points.\n";
			myfile << "Mean x: " << mean_val(0) << " y: " << mean_val(1) << " z: " << mean_val(2) << std::endl << std::endl;
			ROS_INFO("Cluster %d of class %d consists of %d points", instance_id, class_id, (int)cloud_cluster->points.size());
 			ROS_INFO("Mean x: %f, y: %f, z: %f", mean_val(0), mean_val(1), mean_val(2));

			/* Publish the selected instance */
			if(class_id == global_class_id && instance_id == global_instance_id)
			{
				pcl::PointXYZ targetPoint;
				Eigen::Vector2f target_coefficients;
				Eigen::Vector2f initial_angle;
				bool target_found;
				/* For the selected instance (if it's a chair) find the front */
				if(class_id == 0)
				{
					/* Find a vertical plane */
					pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
					pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	 					
	  				// Create the segmentation object
	  				pcl::SACSegmentation<pcl::PointXYZRGB> seg;
	  				seg.setOptimizeCoefficients (true);
	  				seg.setModelType (pcl::SACMODEL_PARALLEL_PLANE);
	  				seg.setMethodType (pcl::SAC_RANSAC);
	  				seg.setDistanceThreshold (0.05);
	  				seg.setAxis (Eigen::Vector3f (0.0, 0.0, 1.0));
					seg.setEpsAngle (0.4); //+-10deg
					seg.setMaxIterations(200);

	  				seg.setInputCloud(cloud_cluster);
	  				seg.segment (*inliers, *coefficients);

					/* Split the general cloud into class clouds */
					find_chair_direction(cloud_cluster, coefficients);

					initial_angle(0) = coefficients->values[0];
					initial_angle(1) = coefficients->values[1];

					target_found = find_clear_point(cloud_cluster, initial_angle, mean_val, &targetPoint, 0.5, 1.5);
				}
				else
				{
					initial_angle(0) = 0;
					initial_angle(1) = 1;
					target_found = find_clear_point_arc(cloud_cluster, initial_angle, mean_val, &targetPoint, M_PI, 1.5);
				}

				/* Calculate target direction vector */
				double x = mean_val(0)-targetPoint.x;
				double y = mean_val(1)-targetPoint.y;
				target_coefficients(0) = x/sqrt(pow(x,2)+pow(y,2));
				target_coefficients(1) = y/sqrt(pow(x,2)+pow(y,2));

				if(target_found && global_start_moving)
				{
					geometry_msgs::PoseStamped goal_pose;

					goal_pose.header.stamp = header.stamp;
					goal_pose.header.frame_id = header.frame_id;

					goal_pose.pose.position.x = targetPoint.x;
					goal_pose.pose.position.y = targetPoint.y;
					goal_pose.pose.position.z = 0;

					goal_pose.pose.orientation = tf::createQuaternionMsgFromYaw(asin(targetPoint.y));

					pub_goal_position.publish(goal_pose);
					global_start_moving = false;
				}

			    /* Set cloud parameters redo as points were added */
			    cloud_cluster->width = cloud_cluster->points.size();
			    cloud_cluster->height = 1;
			    cloud_cluster->is_dense = true;

				pcl::toROSMsg(*cloud_cluster,pcl_message);
				/* Copy the header */
				pcl_message.header.frame_id = header.frame_id;
				pcl_message.header.stamp = header.stamp;
				  	
				/* Publish the pointcloud */
				pub_instance.publish(pcl_message);

				/* Create a cloud  */
	    		pcl::PointCloud<pcl::PointXYZRGB>::Ptr targetpoint_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
	    		pcl::PointXYZRGB pnt;
				float radius = 0.01; 
				float px, py;
				for (float phi=0; phi < M_PI; phi+=M_PI/50)
				{ 
	        		for (float theta=0; theta<2*M_PI;theta+=2*M_PI/50)
	        		{ 
		                px = radius*sin(phi)*cos(theta)+targetPoint.x; 
		                py = radius*sin(phi)*sin(theta)+targetPoint.y; 
		                pnt.x = px; 
		                pnt.y = py; 
		                pnt.z = 0; 
	 					pnt.r = 0;
	 					pnt.g = 255;
	 					pnt.b = 0;
		                targetpoint_cluster->points.push_back(pnt); 
		            }
        		}

        		/* Show line */
			for(double distfromstart=0.0;distfromstart<0.5;distfromstart+=0.02)
			{  						
		  		pnt.x = targetPoint.x + distfromstart*target_coefficients(0);
				pnt.y = targetPoint.y + distfromstart*target_coefficients(1);
				pnt.z = 0;
	 			pnt.r = 0;
	 			pnt.g = 255;
	 			pnt.b = 0;
		  		targetpoint_cluster->points.push_back(pnt);
			}	
        		/* Set cloud parameters redo as points were added */
			targetpoint_cluster->width = targetpoint_cluster->points.size();
			targetpoint_cluster->height = 1;
			targetpoint_cluster->is_dense = true;

			pcl::toROSMsg(*targetpoint_cluster,pcl_message);
			/* Copy the header */
			pcl_message.header.frame_id = header.frame_id;
			pcl_message.header.stamp = header.stamp;
				  	
			/* Publish the pointcloud */
			pub_target.publish(pcl_message);
			}
	  	}

	  	myfile << std::endl << std::endl;
	}


	/* Close results file */
	myfile.close();
  	
}

/* Callback for class id setting */
void callback_class(const std_msgs::UInt16::ConstPtr& class_id)
{
 	global_class_id = (uint16_t)class_id->data;
 	ROS_INFO("Recieved global class id: %d", global_class_id);
}

/* Callback for instance id setting */
void callback_instance(const std_msgs::UInt16::ConstPtr& instance_id)
{
	global_instance_id = (uint16_t)instance_id->data;
	ROS_INFO("Recieved global instance id: %d", global_instance_id);
}

void callback_map(const nav_msgs::OccupancyGrid::ConstPtr& msg)
{
    ROS_INFO("Setting global grid map");
    ROS_INFO("Width %d height %d", msg->info.width, msg->info.height);
    global_grid_map.setMap(msg);
    ROS_INFO("Map set");
}

void callback_start(const std_msgs::Bool::ConstPtr& start_allowed)
{
	global_start_moving = (bool)start_allowed->data;
	ROS_INFO("Recieved start going msg");
}

int main(int argc, char** argv)
{
  	ros::init(argc, argv, "instance_extractor");
  	ros::NodeHandle nh;

  	/* Create subsribers */
	//ros::Subscriber sub = nh.subscribe("/map_assembler/cloud_map", 1, callback);
	ros::Subscriber sub = nh.subscribe("/map_assembler/cloud_map", 1, callback);
	ros::Subscriber sub_class = nh.subscribe("/set_class", 1, callback_class);
	ros::Subscriber sub_instance = nh.subscribe("/set_instance", 1, callback_instance);
	ros::Subscriber sub_cost_map = nh.subscribe("/octomap_grid", 1, callback_map);
	ros::Subscriber sub_start = nh.subscribe("/start_going", 1, callback_start);

	/* Create publisher */
	pub_instance = nh.advertise<sensor_msgs::PointCloud2> ("/instance_model", 1);
	pub_target = nh.advertise<sensor_msgs::PointCloud2> ("/target_point", 1);
	pub_goal_position = nh.advertise<geometry_msgs::PoseStamped> ("move_base_simple/goal", 1);

	ROS_INFO("Started processing - clustering pcl");

	ros::spin();
	return 0;
}
